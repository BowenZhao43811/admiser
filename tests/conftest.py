"""
Shared helpers for the ADMISER test suite.

`conftest.py` is a special file name: pytest imports it automatically and makes
everything defined here available to every test file in this directory. Test
files therefore do NOT need to import from it -- fixtures are requested simply by
naming them as a test function argument, and plain helper functions are imported
normally.
"""

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# The examples shipped with the package. Every one of them is exercised by the
# gradient test, so adding a new example here automatically puts it under test.
# ---------------------------------------------------------------------------
EXAMPLE_NAMES = [
    "my_bang_bang",
    "my_bitumen_pyrolysis",
    "my_euler_buckling_beam",
    "my_ex_882_zeta_param",
    "my_free_terminal_time",
    "my_free_terminal_time2",
    "my_medical1",
    "my_medical2",
    "my_policy_param_problem",
    "my_port_kobe",
    "my_robot_problem",
    "my_state_constrained_problem",
    "my_state_control_constraint",
]


@pytest.fixture(scope="session")
def example_problems():
    """
    Import every example once and hand back {name: problem}.

    `scope="session"` means this runs a single time for the whole test run
    instead of once per test, which matters here because importing an example
    module builds its OCPProblem.
    """
    return {name: importlib.import_module(f"admiser.examples.{name}").problem
            for name in EXAMPLE_NAMES}


def perturbed_point(problem, seed=0, scale=0.01):
    """
    A reproducible point near the problem's initial guess.

    Derivatives are checked away from the initial guess on purpose: an initial
    guess is often a flat, symmetric point (all zeros, or a box midpoint) where a
    wrong gradient can accidentally still look right. `seed` fixes the random
    numbers so a failure is always reproducible.
    """
    z0 = problem.initial_guess()
    rng = np.random.default_rng(seed)
    return z0 + scale * rng.standard_normal(z0.size)


def central_difference_jacobian(fun, z, h=1e-6):
    """
    Approximate the Jacobian of `fun` at `z` by central differences.

    For each variable i it evaluates fun(z + h*e_i) and fun(z - h*e_i) and takes

        dfun/dz_i ~ [fun(z + h*e_i) - fun(z - h*e_i)] / (2h)

    This is the independent reference the AD gradient is compared against: it
    uses nothing but function values, so it shares no code with the AD tape.

    Returns an array of shape (len(fun(z)), len(z)).
    """
    z = np.asarray(z, dtype=float)
    f0 = np.atleast_1d(np.asarray(fun(z), dtype=float))
    jac = np.zeros((f0.size, z.size))
    for i in range(z.size):
        z_plus = z.copy()
        z_plus[i] += h
        z_minus = z.copy()
        z_minus[i] -= h
        f_plus = np.atleast_1d(np.asarray(fun(z_plus), dtype=float))
        f_minus = np.atleast_1d(np.asarray(fun(z_minus), dtype=float))
        jac[:, i] = (f_plus - f_minus) / (2.0 * h)
    return jac


def relative_error(a, b):
    """
    Largest entrywise difference between `a` and `b`, scaled by the magnitude of
    `b`.

    The denominator is max(1, max|b|) rather than max|b| so that a reference of
    almost zero cannot blow the ratio up: near zero we simply fall back to
    comparing absolute differences.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))
