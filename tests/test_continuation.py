"""
Test 6: the eps -> 0 continuation.

A continuation solve starts from a loose approximation of the path constraints
and tightens it round by round: eps (and gamma with it) shrinks by a fixed factor,
and every round starts from the previous round's solution. It stops at the first
round that fails and returns the last round that converged. So the interesting
cases are not only "everything worked", but every way a round can fail -- and
what the user is told when it does.

Forcing a failure on demand
---------------------------
SLSQP does not fail on cue, so some tests below wrap SciPy's minimize and
overwrite the result of one chosen round with a failure. Everything else --
compiling, solving, reporting -- runs for real.

Run just this file with:

    pytest tests/test_continuation.py -v
"""

import warnings

import numpy as np
import jax.numpy as jnp
import pytest

import admiser.sqp_solver as sqp_solver
from admiser import OCPProblem, OCPSolver, make_builders
from admiser.sqp_solver import (STATUS_CONVERGED, STATUS_MAX_ITERATIONS,
                                STATUS_NO_ROUND_SUCCEEDED, STATUS_SLSQP_FAILURE)

# The toy problem below runs over a horizon of T = 1.
HORIZON = 1.0


def _toy_problem(eps=0.5, gamma=None, mode="continuation", n_rounds=3, shrink=0.1):
    """
    x' = u, x(0) = 0. The objective pulls x towards 1, the path constraint
    x(t) <= 0.5 stops it halfway -- so the constraint is active and the
    continuation has real work to do.
    """
    def dyn(x, u, theta=None):
        return jnp.array([u[0]])

    def L(t, x, u, theta):
        return (x[0] - 1.0) ** 2 + 0.1 * u[0] * u[0]

    p = OCPProblem(
        N=8, dt=HORIZON / 8,
        x0=np.array([0.0]), u0=0.0,
        dyn=dyn, m_sub=2,
        nu=1, nx=1,
        objective_builder=make_builders(dyn=dyn, L=L, Phi=None),
        control_bounds_builder=lambda q: [(-5.0, 5.0)] * (q.N * q.nu),
    )
    p.add_path_ineq(lambda t, x, u, th: x[0] - 0.5, eps=eps, gamma=gamma)
    p.set_transcription(mode=mode, n_rounds=n_rounds, shrink=shrink)
    return p


def _fail_round(monkeypatch, which, status=8, message="forced failure for this test"):
    """Make SLSQP report a failure in round `which` (counting from 1); the rest are real."""
    real_minimize = sqp_solver.minimize
    calls = {"n": 0}

    def minimize(*args, **kwargs):
        res = real_minimize(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == which:
            res.success, res.status, res.message = False, status, message
        return res

    monkeypatch.setattr(sqp_solver, "minimize", minimize)


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------
def test_the_registered_eps_is_the_first_round():
    """eps is a STARTING value: the rounds run eps, eps*s, eps*s^2, ..."""
    p = _toy_problem(eps=0.5, n_rounds=3, shrink=0.1)
    eps = [float(e[0]) for e, _g in p.transcription_rounds()]
    np.testing.assert_allclose(eps, [0.5, 0.05, 0.005])


def test_an_automatic_gamma_follows_eps():
    """Without an explicit gamma, every round uses gamma = T*eps/4."""
    p = _toy_problem(eps=0.5)
    for e, g in p.transcription_rounds():
        assert g[0] == pytest.approx(HORIZON * e[0] / 4.0)


def test_an_explicit_gamma_shrinks_with_eps():
    """An explicit gamma is the first round's value and shrinks by the same factor."""
    p = _toy_problem(eps=0.5, gamma=0.01, n_rounds=3, shrink=0.1)
    gammas = [float(g[0]) for _e, g in p.transcription_rounds()]
    np.testing.assert_allclose(gammas, [0.01, 0.001, 0.0001])


def test_single_mode_is_one_round_at_the_registered_eps():
    p = _toy_problem(eps=0.5, mode="single")
    rounds = p.transcription_rounds()
    assert len(rounds) == 1
    assert float(rounds[0][0][0]) == 0.5


def test_to_nlp_uses_the_registered_eps():
    """to_nlp() hands back the first round's transcription, i.e. the registered values."""
    nlp = OCPSolver(_toy_problem(eps=0.5)).to_nlp()
    np.testing.assert_allclose(nlp.eps, [0.5])
    np.testing.assert_allclose(nlp.gamma, [HORIZON * 0.5 / 4.0])


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------
def test_a_clean_continuation_reports_status_0_and_no_warning():
    p = _toy_problem(eps=0.5, n_rounds=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")         # any warning fails this test
        res = OCPSolver(p).solve(maxiter=500, ftol=1e-10, verbose=False)

    assert res["status"] == STATUS_CONVERGED and res["success"]
    assert len(res["rounds"]) == 3
    assert all(r["success"] for r in res["rounds"])
    # the solution returned is the last, tightest round
    assert [r["returned"] for r in res["rounds"]] == [False, False, True]
    np.testing.assert_allclose(res["final_eps"], [0.005])
    assert res["J_opt"] == res["rounds"][-1]["J_opt"]


def test_tightening_eps_tightens_the_constraint():
    """
    From inexact to exact: the loose first round may overshoot the constraint,
    and every later round must violate it no more than the one before.
    """
    res = OCPSolver(_toy_problem(eps=0.5, n_rounds=3)).solve(maxiter=500, ftol=1e-10,
                                                             verbose=False)
    viol = [r["max_path_viol"] for r in res["rounds"]]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(viol, viol[1:])), viol


def test_one_compilation_serves_every_round(monkeypatch):
    """
    eps and gamma are passed into the compiled NLP as arguments, so the rounds of
    a continuation must not rebuild it.
    """
    calls = {"n": 0}
    real_build = sqp_solver.build_nlp

    def counting_build(problem):
        calls["n"] += 1
        return real_build(problem)

    monkeypatch.setattr(sqp_solver, "build_nlp", counting_build)
    OCPSolver(_toy_problem(n_rounds=3)).solve(maxiter=500, verbose=False)
    assert calls["n"] == 1


def test_a_failed_later_round_returns_the_last_converged_round(monkeypatch):
    _fail_round(monkeypatch, which=2)
    with pytest.warns(RuntimeWarning, match="last round that converged"):
        res = OCPSolver(_toy_problem(n_rounds=3)).solve(maxiter=500, verbose=False)

    assert res["status"] == STATUS_SLSQP_FAILURE and not res["success"]
    assert len(res["rounds"]) == 2, "rounds after the failure must not be run"
    assert res["rounds"][1]["scipy_status"] == 8, "SciPy's own status is kept per round"
    assert [r["returned"] for r in res["rounds"]] == [True, False]
    np.testing.assert_allclose(res["final_eps"], [0.5])
    assert res["J_opt"] == res["rounds"][0]["J_opt"]


def test_running_out_of_iterations_adopts_the_iterate():
    """
    A round that only hit maxiter is adopted -- its iterate is usually far better
    than where it started -- but it is reported as NOT converged.
    """
    p = _toy_problem(n_rounds=3)
    with pytest.warns(RuntimeWarning, match="NOT converged"):
        res = OCPSolver(p).solve(maxiter=1, verbose=False)

    assert res["status"] == STATUS_MAX_ITERATIONS
    assert len(res["rounds"]) == 1
    assert res["rounds"][0]["returned"]
    assert not np.allclose(res["scipy_result"].x, p.initial_guess())


def test_a_failed_first_round_returns_its_iterate_not_the_initial_guess(monkeypatch):
    """
    Nothing converged, so there is no good answer -- but the iterate SLSQP reached
    is returned, flagged, rather than silently falling back to the initial guess.
    """
    _fail_round(monkeypatch, which=1)
    p = _toy_problem(n_rounds=3)
    with pytest.warns(RuntimeWarning, match="first round already failed"):
        res = OCPSolver(p).solve(maxiter=500, verbose=False)

    assert res["status"] == STATUS_NO_ROUND_SUCCEEDED
    assert len(res["rounds"]) == 1
    assert not np.allclose(res["scipy_result"].x, p.initial_guess())


def test_a_failed_single_solve_is_flagged_too(monkeypatch):
    """Single mode is simply a one-round continuation, with the same reporting."""
    _fail_round(monkeypatch, which=1)
    with pytest.warns(RuntimeWarning):
        res = OCPSolver(_toy_problem(mode="single")).solve(maxiter=500)
    assert res["status"] == STATUS_NO_ROUND_SUCCEEDED and not res["success"]
