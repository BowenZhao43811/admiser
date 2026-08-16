"""
Test 2: every quadrature scheme must actually deliver the order it advertises.

QUAD_SCHEMES declares an order for each scheme (1, 1, 2, 2, 3, 4). Those numbers
are not decoration -- users pick a scheme by them, and an earlier version of this
package shipped a scheme that claimed to match RK4 while quietly delivering only
2nd order, which made every reported objective value systematically too low.

An order claim is testable. Halve the step size and watch how fast the error
falls: an order-p rule divides its error by 2^p. So we solve a problem whose exact
answer is known, refine the step, and read the order straight off the errors.

Run just this file with:

    pytest tests/test_quadrature_order.py -v
"""

import numpy as np
import pytest
from functools import partial

from admiser import (OCPProblem, OCPSolver, QUAD_SCHEMES, make_builders,
                     rk4_substeps)
from admiser.ocp_integrators_utils import _quad_samples

# ---------------------------------------------------------------------------
# Reference problem with a closed-form answer
#
#     x' = x,  x(0) = 1          =>  x(t) = e^t
#     g(t, x) = t^2 * x
#     int_0^1 t^2 e^t dt = e - 2  (exactly)
#
# The integrand depends on BOTH t and x. A g that depended only on t would let a
# scheme hide an inaccurate state estimate at its sample points, which is exactly
# the mistake this test exists to catch.
# ---------------------------------------------------------------------------
EXACT_INTEGRAL = np.e - 2.0

# Substep counts used for the refinement study. Each entry halves the step size
# relative to the previous one.
SUBSTEP_COUNTS = (1, 2, 4, 8, 16)

# How far the measured order may sit from the declared one. The measurement is
# itself approximate -- the asymptotic rate is only reached as h -> 0, and at some
# point round-off puts a floor under the error -- so a band rather than an exact
# match is the right check. Measured values are 0.98, 1.02, 2.00, 2.00, 2.96, 4.00.
ORDER_TOLERANCE = 0.25


def _dyn(x, u, theta=None):
    """x' = x"""
    return np.array([x[0]], dtype=object)


def _integrand(t, x, u, theta):
    """g(t, x) = t^2 * x"""
    return t * t * x[0]


def _integral_with(scheme, m_sub):
    """
    Compute int_0^1 t^2 e^t dt with the given scheme and substep count.

    The integral is obtained by making it this problem's objective: the objective
    is exactly int L dt, so the value the tape reports IS the quadrature result.
    """
    objective_builder = make_builders(dyn=_dyn, L=_integrand, Phi=None, quad=scheme)
    problem = OCPProblem(
        N=4, dt=0.25,
        x0=np.array([1.0]),
        u0=0.0,
        dyn=_dyn,
        integrator=partial(rk4_substeps, m_sub=m_sub),
        nu=1, nx=1,
        objective_builder=objective_builder,
        control_bounds_builder=lambda p: [(-1.0, 1.0)] * (p.N * p.nu),
        ntheta=0,
    )
    # to_nlp() records the tape without optimising; the control does not enter
    # this integral at all, so no solve is needed.
    nlp = OCPSolver(problem).to_nlp()
    return nlp.objective_fun(problem.initial_guess())


def _measured_order(errors):
    """
    Estimate the order of convergence from a sequence of errors.

    Halving the step multiplies the error by 2^(-p), so

        p ~ log2(error_before / error_after)

    The last two ratios are averaged because the early, coarse steps have not
    reached the asymptotic regime yet.
    """
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    return float(np.mean(rates[-2:]))


@pytest.mark.parametrize("scheme", list(QUAD_SCHEMES))
def test_scheme_reaches_its_declared_order(scheme):
    """
    Refine the substep size and check the error falls at the declared rate.
    """
    declared = QUAD_SCHEMES[scheme].order

    errors = [abs(_integral_with(scheme, m) - EXACT_INTEGRAL)
              for m in SUBSTEP_COUNTS]

    # Errors must shrink monotonically; if they do not, the refinement study
    # itself is meaningless and the order estimate below would be noise.
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)), (
        f"{scheme}: refining the step did not reduce the error: {errors}"
    )

    measured = _measured_order(errors)
    assert abs(measured - declared) < ORDER_TOLERANCE, (
        f"{scheme}: declares order {declared} but measures {measured:.2f}. "
        f"errors at m_sub={SUBSTEP_COUNTS}: "
        + ", ".join(f"{e:.2e}" for e in errors)
    )


@pytest.mark.parametrize("scheme", list(QUAD_SCHEMES))
def test_sample_weights_sum_to_the_substep_length(scheme):
    """
    Every scheme integrates over the same span, so its weights must add up to h.

    This is the cheapest possible guard against a whole class of mistakes: get
    the weights wrong and every integral in the package comes out scaled by a
    constant factor -- an error that is invisible in a plot but changes every
    objective value.
    """
    h = 0.37
    x = np.array([1.0])
    k1, k2, k3 = np.array([2.0]), np.array([2.1]), np.array([2.2])
    x_end = np.array([1.7])

    samples = _quad_samples(scheme, 0.0, h, x, k1, k2, k3, x_end)
    total_weight = sum(w for (_t, _x, w) in samples)

    assert abs(total_weight - h) < 1e-15, (
        f"{scheme}: weights sum to {total_weight!r}, expected the substep length {h!r}"
    )


@pytest.mark.parametrize("scheme", list(QUAD_SCHEMES))
def test_number_of_integrand_evaluations_is_as_declared(scheme):
    """
    n_eval is what users budget their AD tape size against, so it must be honest.
    """
    h = 0.37
    x = np.array([1.0])
    k1, k2, k3 = np.array([2.0]), np.array([2.1]), np.array([2.2])
    x_end = np.array([1.7])

    samples = _quad_samples(scheme, 0.0, h, x, k1, k2, k3, x_end)
    assert len(samples) == QUAD_SCHEMES[scheme].n_eval, (
        f"{scheme}: produces {len(samples)} sample points but declares "
        f"n_eval={QUAD_SCHEMES[scheme].n_eval}"
    )


def test_quadrature_scheme_does_not_change_the_trajectory():
    """
    The scheme controls how integrals are sampled, nothing else. The state is
    always advanced by full RK4, so the end state must be bit-for-bit identical
    across every scheme. If this ever fails, a scheme has started interfering
    with the dynamics.
    """
    end_states = [
        float(rk4_substeps(np.array([1.0]), np.array([0.0]), 0.5,
                           lambda x, u: np.array([x[0]]),
                           m_sub=7, accumulate_cb=lambda *args: None,
                           t0=0.0, quad=scheme)[0])
        for scheme in QUAD_SCHEMES
    ]
    assert len(set(end_states)) == 1, (
        f"the end state depends on the quadrature scheme, which it must not: {end_states}"
    )


def test_unknown_scheme_names_are_rejected():
    """
    A misspelled scheme name must raise, never fall through to a default.

    An unrecognised name used to make the integrator skip its callback entirely,
    so the objective silently evaluated to zero -- the worst kind of failure,
    because it looks like a converged answer.
    """
    for bad_name in ("rk4-mid", "mid", "RK4", "simpsons", ""):
        with pytest.raises(ValueError):
            rk4_substeps(np.array([1.0]), np.array([0.0]), 0.5,
                         lambda x, u: np.array([x[0]]),
                         m_sub=2, quad=bad_name)
