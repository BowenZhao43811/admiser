"""
Test 1: the AD derivatives must agree with finite differences.

This is the central claim of the whole package. ADMISER replaces the continuous
adjoint equations of Teo's control parametrization with an automatic-differentiation
tape, and the optimiser then trusts whatever that tape reports. If a refactor ever
breaks the tape, the solver will not crash -- it will happily converge to the wrong
answer, guided by wrong derivatives, and print a perfectly plausible number.

So we check the derivatives against a completely independent reference: central
finite differences, which use nothing but function values.

Run just this file with:

    pytest tests/test_ad_gradients.py -v
"""

import numpy as np
import pytest

from admiser import OCPSolver

from conftest import (EXAMPLE_NAMES, central_difference_jacobian,
                      perturbed_point, relative_error)

# How close AD and finite differences must agree.
#
# The limit here is the REFERENCE, not the AD. Central differences with a step of
# 1e-6 carry a truncation error of order h^2 = 1e-12 plus a round-off error of
# order eps_machine/h = 1e-10, and both get amplified by the conditioning of the
# problem. Measured across all examples the worst disagreement is about 1.3e-07,
# so 1e-5 leaves roughly two orders of headroom: tight enough that a genuinely
# wrong derivative cannot slip through, loose enough not to fail spuriously.
GRADIENT_TOLERANCE = 1e-5


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_objective_gradient_matches_finite_difference(name, example_problems):
    """
    dJ/dz from the AD tape must match central differences of J(z).

    `@pytest.mark.parametrize` runs this function once per example and reports
    each as its own PASS/FAIL line, so a failure names the guilty example
    immediately instead of just saying "gradients are broken".
    """
    problem = example_problems[name]

    # to_nlp() records the AD tape and returns the transcribed NLP WITHOUT
    # running the optimiser -- exactly what we want, since we are testing
    # derivatives, not convergence. For a problem configured for continuation it
    # uses the eps of the first round.
    nlp = OCPSolver(problem).to_nlp()

    z = perturbed_point(problem, seed=0)

    grad_ad = np.atleast_2d(nlp.objective_grad(z))
    grad_fd = central_difference_jacobian(nlp.objective_fun, z)

    err = relative_error(grad_ad, grad_fd)
    assert err < GRADIENT_TOLERANCE, (
        f"{name}: AD objective gradient disagrees with finite differences "
        f"by a relative {err:.2e} (tolerance {GRADIENT_TOLERANCE:.0e})"
    )


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_constraint_jacobians_match_finite_difference(name, example_problems):
    """
    The same check for the constraint Jacobians dG/dz and dC/dz.

    Constraint derivatives are just as load-bearing as the objective's: SLSQP
    uses them to decide which way is feasible. Examples without constraints are
    skipped rather than silently passing, so the report tells the truth about
    what was actually verified.
    """
    problem = example_problems[name]
    nlp = OCPSolver(problem).to_nlp()
    z = perturbed_point(problem, seed=0)

    checked = []

    if nlp.eq_ad is not None:
        err = relative_error(nlp.eq_jac(z),
                             central_difference_jacobian(nlp.eq_fun, z))
        assert err < GRADIENT_TOLERANCE, (
            f"{name}: AD equality Jacobian disagrees with finite differences "
            f"by a relative {err:.2e}"
        )
        checked.append("eq")

    if nlp.ineq_ad is not None:
        err = relative_error(nlp.ineq_jac(z),
                             central_difference_jacobian(nlp.ineq_fun, z))
        assert err < GRADIENT_TOLERANCE, (
            f"{name}: AD inequality Jacobian disagrees with finite differences "
            f"by a relative {err:.2e}"
        )
        checked.append("ineq")

    if not checked:
        pytest.skip(f"{name} has no equality or inequality constraints")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gradient_holds_at_several_points(seed, example_problems):
    """
    A wrong gradient can look right at one unlucky point, so re-check at a few
    more.

    Only one example is used here -- it has both equality and inequality
    constraints and system parameters, so it exercises every branch of the tape
    builder -- which keeps this extra confidence cheap.
    """
    problem = example_problems["my_euler_buckling_beam"]
    nlp = OCPSolver(problem).to_nlp()
    z = perturbed_point(problem, seed=seed, scale=0.02)

    err = relative_error(np.atleast_2d(nlp.objective_grad(z)),
                         central_difference_jacobian(nlp.objective_fun, z))
    assert err < GRADIENT_TOLERANCE, f"seed={seed}: relative error {err:.2e}"


def test_the_comparison_would_actually_catch_a_wrong_gradient(example_problems):
    """
    Who tests the test?

    A comparison that always passes is worthless, and it is surprisingly easy to
    write one by accident (comparing something to itself, an unreachable branch,
    a tolerance so loose nothing can fail). Here we deliberately corrupt a
    gradient and assert that the comparison rejects it. If this test ever fails,
    the other tests in this file have stopped proving anything.
    """
    problem = example_problems["my_bang_bang"]
    nlp = OCPSolver(problem).to_nlp()
    z = perturbed_point(problem, seed=0)

    grad_fd = central_difference_jacobian(nlp.objective_fun, z)
    grad_ad = np.atleast_2d(nlp.objective_grad(z))

    # sanity: the honest comparison passes
    assert relative_error(grad_ad, grad_fd) < GRADIENT_TOLERANCE

    # now break one single entry by a tiny amount and make sure we notice
    corrupted = grad_ad.copy()
    corrupted[0, 0] += 1e-3
    assert relative_error(corrupted, grad_fd) > GRADIENT_TOLERANCE, (
        "the gradient comparison failed to notice a corrupted entry, "
        "so it is not actually testing anything"
    )
