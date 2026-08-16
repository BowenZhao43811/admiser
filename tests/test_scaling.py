"""
Test 4: automatic problem scaling.

Scaling changes how the optimiser SEES the problem, so the thing that matters
most is that it does not change the problem itself: the same minimiser, the same
feasible set, and every reported number back in the user's own units. A scaling
bug would not raise -- it would quietly report numbers in the wrong units, or
stop the solver at the wrong place.

Run just this file with:

    pytest tests/test_scaling.py -v
"""

import importlib
from functools import partial

import numpy as np
import pytest

from admiser import OCPProblem, OCPSolver, make_builders, rk4_substeps
from admiser.problem_scaling import compute_scaling, identity_scaling


def _toy_problem(objective_magnitude=1.0):
    """
    x' = u with a quadratic cost, multiplied by `objective_magnitude` so the same
    problem can be presented at wildly different scales.
    """
    def dyn(x, u, theta=None):
        return np.array([u[0]], dtype=object)

    def L(t, x, u, theta):
        return objective_magnitude * (x[0]*x[0] + u[0]*u[0])

    return OCPProblem(
        N=8, dt=0.125,
        x0=np.array([1.0]), u0=0.0,
        dyn=dyn,
        integrator=partial(rk4_substeps, m_sub=4),
        nu=1, nx=1,
        objective_builder=make_builders(dyn=dyn, L=L, Phi=None, quad='rk4'),
        control_bounds_builder=lambda p: [(-5.0, 5.0)] * (p.N * p.nu),
        ntheta=0,
    )


def test_scaling_does_not_move_the_minimiser():
    """
    A positive constant on the objective cannot change where the minimum is. The
    same problem written at 1x and at 1e6x must therefore reach the same controls,
    and each must report its own objective in its own units.
    """
    small = _toy_problem(1.0)
    huge = _toy_problem(1e6)

    r_small = OCPSolver(small).solve(maxiter=500, ftol=1e-10, verbose=False)
    r_huge = OCPSolver(huge).solve(maxiter=500, ftol=1e-10, verbose=False)

    np.testing.assert_allclose(r_huge["U_opt"], r_small["U_opt"], rtol=1e-6, atol=1e-8)
    assert r_huge["J_opt"] == pytest.approx(1e6 * r_small["J_opt"], rel=1e-6), (
        "the objective was not reported in the user's own units"
    )


def test_scaling_off_and_on_agree_on_a_well_scaled_problem():
    """
    Where scaling is not needed it must not do harm: a problem that is already of
    order one must give the same answer with scaling on and off.
    """
    on = _toy_problem(1.0)
    off = _toy_problem(1.0)
    off.set_scaling(objective="none", constraints="none")

    r_on = OCPSolver(on).solve(maxiter=500, ftol=1e-10, verbose=False)
    r_off = OCPSolver(off).solve(maxiter=500, ftol=1e-10, verbose=False)

    assert r_on["J_opt"] == pytest.approx(r_off["J_opt"], rel=1e-8)


def test_factors_are_deterministic():
    """
    The factors are estimated by probing points around the initial guess. Those
    probes must be fixed offsets, never random: otherwise two runs of the same
    script would optimise differently scaled problems and their published numbers
    would not be comparable.
    """
    first = OCPSolver(_toy_problem(1e3)).solve(maxiter=50, verbose=False)["scaling"]
    second = OCPSolver(_toy_problem(1e3)).solve(maxiter=50, verbose=False)["scaling"]

    assert first.objective == second.objective
    np.testing.assert_array_equal(first.eq, second.eq)
    np.testing.assert_array_equal(first.ineq, second.ineq)


def test_scaling_none_is_the_identity():
    p = _toy_problem(1e6)
    p.set_scaling(objective="none", constraints="none")
    sc = OCPSolver(p).solve(maxiter=50, verbose=False)["scaling"]
    assert sc.is_identity


def test_an_explicit_factor_is_used_verbatim():
    p = _toy_problem(1e6)
    p.set_scaling(objective=1e-6, constraints="none")
    sc = OCPSolver(p).solve(maxiter=50, verbose=False)["scaling"]
    assert sc.objective == 1e-6
    assert sc.objective_mode == "manual"


def test_bad_scaling_settings_are_rejected():
    for bad in ("Auto", "yes", 0.0, -1.0):
        with pytest.raises(ValueError):
            _toy_problem().set_scaling(objective=bad)
    with pytest.raises(ValueError):
        _toy_problem().set_scaling(constraints="sometimes")


def test_a_degenerate_starting_point_does_not_produce_an_absurd_factor():
    """
    my_port_kobe starts at a point where J(z0) = 0 AND grad J(z0) = 0, so a naive
    1/|J(z0)| would divide by zero. The fallback probe must recover a sane factor
    -- and, just as importantly, a SMALL probe: an early version stepped a quarter
    of the way across the box, landed in a region where the dynamics blow up, and
    produced a factor of ~1e-8 that made ftol so loose the solver stopped at the
    initial guess.
    """
    EX = importlib.import_module("admiser.examples.my_port_kobe")
    p = EX.problem
    z0 = p.initial_guess()

    unscaled = OCPSolver(p).to_nlp()
    assert unscaled.objective_fun(z0) == 0.0, "this test assumes a degenerate start"

    sc = compute_scaling(unscaled, z0, p.make_bounds())
    assert np.isfinite(sc.objective) and sc.objective > 0
    # A factor pinned at the clamp limit would mean the estimate was nonsense.
    assert 1e-6 < sc.objective < 1e6, f"implausible objective factor {sc.objective:.3e}"


def test_to_nlp_is_unscaled():
    """
    to_nlp() is meant to hand back the user's own problem, so a gradient checked
    against finite differences there is a gradient of what the user wrote.
    """
    p = _toy_problem(1e6)
    nlp = OCPSolver(p).to_nlp()
    assert nlp.scaling.is_identity

    z0 = p.initial_guess()
    # the value must be the raw, unscaled objective
    solver = OCPSolver(p)
    solver.solve(maxiter=20, verbose=False)
    assert nlp.objective_fun(z0) == pytest.approx(1e6 * OCPSolver(_toy_problem(1.0))
                                                  .to_nlp().objective_fun(z0), rel=1e-12)


@pytest.mark.parametrize("name", ["my_medical1", "my_medical2"])
def test_the_medical_examples_now_converge_to_a_feasible_point(name):
    """
    Both used to stop with SLSQP status 8 at an infeasible point, because their
    objectives are far from order one and ftol is an absolute test. my_medical1
    halted after 9 iterations with its dose budget violated by 1e-2.

    Note that neither example carries a hand-written scale factor: they state
    their objectives in natural units and rely on the automatic scaling.
    """
    p = importlib.reload(importlib.import_module(f"admiser.examples.{name}")).problem
    r = OCPSolver(p).solve(maxiter=5000, ftol=1e-12, verbose=False)

    assert r["scipy_result"].status == 0, (
        f"{name} did not converge: {r['scipy_result'].message}"
    )
    violation = max(0.0 if r["ineq_resid"] is None else max(0.0, -np.min(r["ineq_resid"])),
                    0.0 if r["eq_resid"] is None else np.max(np.abs(r["eq_resid"])))
    assert violation < 1e-9, f"{name} converged to an infeasible point, violation {violation:.2e}"


def test_reported_residuals_are_in_user_units():
    """
    Constraint rows are scaled individually, so eq_resid / ineq_resid must be
    divided back. If they were not, a user comparing them against their own
    tolerance would be comparing against the wrong numbers.
    """
    EX = importlib.reload(importlib.import_module("admiser.examples.my_port_kobe"))
    p = EX.problem
    # Single mode on purpose. Under continuation the last round uses the registered
    # (final) eps while to_nlp() tapes with the FIRST round's eps, so the two would
    # be evaluating genuinely different constraints -- a difference of 100x here,
    # since gamma = T*eps/4 -- and the comparison would say nothing about units.
    p.set_transcription(mode="single")
    scaled = OCPSolver(p).solve(maxiter=3000, ftol=1e-10, verbose=False)

    # Re-evaluate the constraints at the solution through an UNSCALED view.
    nlp = OCPSolver(p).to_nlp()
    z = scaled["scipy_result"].x
    np.testing.assert_allclose(scaled["eq_resid"], nlp.eq_fun(z), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(scaled["ineq_resid"], nlp.ineq_fun(z), rtol=1e-9, atol=1e-12)


def test_history_is_in_user_units():
    """history_cost feeds convergence plots, so it must match J_opt's units."""
    p = _toy_problem(1e6)
    r = OCPSolver(p).solve(maxiter=200, ftol=1e-10, verbose=False)
    assert r["history_cost"].size > 0
    assert r["history_cost"][-1] == pytest.approx(r["J_opt"], rel=1e-6)


def test_identity_scaling_helper():
    sc = identity_scaling(3, 2)
    assert sc.is_identity
    assert sc.eq.shape == (3,) and sc.ineq.shape == (2,)
    assert sc.objective_to_user(5.0) == 5.0
