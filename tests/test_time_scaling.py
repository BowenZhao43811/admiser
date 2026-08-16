"""
Test 3: the time-scaling transform (CPET).

Turning the segment durations into decision variables touches almost every part
of the bookkeeping -- the layout of z, the box bounds, the time grid, the horizon
used by auto_gamma -- so most of what is checked here is that those pieces still
agree with one another. The derivatives through the transform are already covered
by test_ad_gradients.py, because both free-terminal-time examples use it and are
listed in EXAMPLE_NAMES.

Run just this file with:

    pytest tests/test_time_scaling.py -v
"""

import importlib
from functools import partial

import numpy as np
import pytest

from admiser import OCPProblem, OCPSolver, make_builders, rk4_substeps


# ---------------------------------------------------------------------------
# A tiny problem used for the structural checks. x' = u, so everything about it
# is easy to predict by hand.
# ---------------------------------------------------------------------------
def _toy_problem(N=5, dt=0.2, ntheta=0):
    def dyn(x, u, theta=None):
        return np.array([u[0]], dtype=object)

    def L(t, x, u, theta):
        return x[0] * x[0] + u[0] * u[0]

    return OCPProblem(
        N=N, dt=dt,
        x0=np.array([1.0]), u0=0.0,
        dyn=dyn,
        integrator=partial(rk4_substeps, m_sub=4),
        nu=1, nx=1,
        objective_builder=make_builders(dyn=dyn, L=L, Phi=None, quad='rk4'),
        control_bounds_builder=lambda p: [(-5.0, 5.0)] * (p.N * p.nu),
        ntheta=ntheta,
        theta0=np.zeros(ntheta) if ntheta else None,
        param_bounds_builder=(lambda p: [(-1.0, 1.0)] * ntheta) if ntheta else None,
    )


def test_decision_vector_grows_by_one_duration_per_segment():
    """z = [U (N*nu) ; theta (ntheta) ; tau (N)], and every consumer must agree."""
    p = _toy_problem(N=5, ntheta=2)
    n_before = p.initial_guess().size
    assert n_before == 5 * 1 + 2

    p.set_time_scaling()
    assert p.initial_guess().size == n_before + p.N
    assert len(p.make_bounds()) == n_before + p.N
    # validate() cross-checks the guess against the bounds, so it fails loudly if
    # only one of the two learned about the new block.
    p.validate()


def test_split_decision_returns_the_three_blocks():
    p = _toy_problem(N=5, ntheta=2)
    p.set_time_scaling(tau0=0.3)
    z = p.initial_guess()

    U, theta, tau = p.split_decision(z)
    assert U.size == 5
    assert theta.size == 2
    assert tau.size == 5
    np.testing.assert_allclose(tau, 0.3)

    # Without the transform the third block is simply absent.
    q = _toy_problem(N=5, ntheta=2)
    _, _, tau_none = q.split_decision(q.initial_guess())
    assert tau_none is None


def test_durations_are_bounded_below_by_zero_by_default():
    """
    Negative durations would run time backwards, so they must be impossible.
    Zero is allowed: collapsing a segment is how the optimiser drops one.
    """
    p = _toy_problem()
    p.set_time_scaling()
    tau_bounds = p.make_bounds()[-p.N:]
    assert all(lo == 0.0 for lo, _hi in tau_bounds)

    with pytest.raises(ValueError):
        _toy_problem().set_time_scaling(tau_min=-0.1)
    with pytest.raises(ValueError):
        _toy_problem().set_time_scaling(tau0=-1.0)
    with pytest.raises(ValueError):
        _toy_problem().set_time_scaling(tau_min=1.0, tau_max=0.5)


def test_uniform_durations_reproduce_the_untransformed_problem():
    """
    The transform must be an identity when the durations sit at the uniform values
    the fixed grid would have used.

    This is the sharpest structural check available: it exercises the whole
    transformed path -- the extra decision block, dt_of_segment returning an
    a_double, the quadrature weights becoming functions of tau -- and demands the
    same number as the plain formulation.
    """
    plain = _toy_problem(N=6, dt=0.25)
    scaled = _toy_problem(N=6, dt=0.25)
    scaled.set_time_scaling(tau0=0.25)

    J_plain = OCPSolver(plain).to_nlp().objective_fun(plain.initial_guess())
    J_scaled = OCPSolver(scaled).to_nlp().objective_fun(scaled.initial_guess())

    assert J_scaled == pytest.approx(J_plain, rel=1e-12), (
        f"the transform changed the objective at uniform durations: "
        f"{J_scaled!r} vs {J_plain!r}"
    )


def test_total_time_constraint_pins_the_horizon():
    """
    total_time=T registers sum(tau) = T, which is exactly
    add_integral_eq(lambda t,x,u,th: 1.0, target=T) since int 1 dt = sum(tau).
    """
    p = _toy_problem(N=5, dt=0.2)
    n_eq_before = len(p.int_eq_specs)
    p.set_time_scaling(tau0=0.2, total_time=1.0)
    assert len(p.int_eq_specs) == n_eq_before + 1

    res = OCPSolver(p).solve(maxiter=500, ftol=1e-11, verbose=False)
    assert np.sum(res["tau_opt"]) == pytest.approx(1.0, abs=1e-8)
    assert res["T_opt"] == pytest.approx(1.0, abs=1e-8)


def test_time_grid_follows_the_optimised_durations():
    """
    t_opt must be the cumulative sum of tau, not a uniform linspace: everything
    plotted against time depends on it.
    """
    p = _toy_problem(N=5, dt=0.2)
    p.set_time_scaling(tau0=0.2, total_time=1.0)
    res = OCPSolver(p).solve(maxiter=500, ftol=1e-11, verbose=False)

    expected = np.concatenate(([0.0], np.cumsum(res["tau_opt"])))
    np.testing.assert_allclose(res["t_opt"], expected, rtol=0, atol=1e-14)
    assert res["t_opt"].size == p.N + 1


def test_solving_twice_gives_the_same_answer():
    """solve() must not leave the problem in a modified state."""
    p = _toy_problem(N=5, dt=0.2)
    p.set_time_scaling(tau0=0.2, total_time=1.0)
    tau0_before = p.time_scaling["tau0"].copy()

    r1 = OCPSolver(p).solve(maxiter=500, ftol=1e-11, verbose=False)
    r2 = OCPSolver(p).solve(maxiter=500, ftol=1e-11, verbose=False)

    assert r1["J_opt"] == r2["J_opt"]
    np.testing.assert_array_equal(p.time_scaling["tau0"], tau0_before)


@pytest.mark.parametrize("name,reference_T", [
    ("my_free_terminal_time", 4.3211735),
    ("my_free_terminal_time2", 46.17131),
])
def test_minimum_time_examples_reproduce_the_hand_transformed_answer(name, reference_T):
    """
    Both examples used to carry an extra state and control to encode the horizon
    by hand -- which is the time-scaling transform written out longhand. Rewritten
    with the built-in transform they must find the same minimum time.

    The reference values come from the hand-transformed formulations, so the
    tolerance is loose enough to allow for a different discretisation of the same
    problem while still being far tighter than any real disagreement would be.
    """
    EX = importlib.import_module(f"admiser.examples.{name}")
    res = OCPSolver(EX.problem).solve(maxiter=3000, ftol=1e-10, verbose=False)

    assert res["scipy_result"].status == 0, (
        f"{name} did not converge: {res['scipy_result'].message}"
    )
    assert np.max(np.abs(res["eq_resid"])) < 1e-7, (
        f"{name}: terminal conditions not met, max|eq| = "
        f"{np.max(np.abs(res['eq_resid'])):.2e}"
    )
    assert res["T_opt"] == pytest.approx(reference_T, rel=1e-4), (
        f"{name}: minimum time {res['T_opt']:.9f} differs from the "
        f"hand-transformed reference {reference_T}"
    )
    # Every duration must respect its box bound.
    lo = EX.problem.time_scaling["tau_min"]
    assert np.all(res["tau_opt"] >= lo - 1e-12)
