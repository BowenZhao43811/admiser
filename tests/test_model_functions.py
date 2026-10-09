"""
Test 5: mistakes in model functions must be reported clearly.

JAX differentiates the model functions by tracing them: it calls them once with
stand-in values ("tracers") that record every operation. Code that needs a
concrete number at that moment -- np.exp, a NumPy object array, an if/else on a
state value -- cannot be traced, and JAX's own error message talks about tracers,
not about what to change in the user's code. ADMISER catches these errors and adds
the fix. These tests make sure that the hint keeps appearing, and that the other
common mistakes are named just as plainly.

Run just this file with:

    pytest tests/test_model_functions.py -v
"""

import numpy as np
import jax.numpy as jnp
import pytest

from admiser import OCPProblem, OCPSolver, make_builders


def _problem(dyn, L=None, **kwargs):
    """A one-state problem around the given dynamics; everything else is trivial."""
    if L is None:
        def L(t, x, u, theta):
            return x[0] * x[0] + u[0] * u[0]
    return OCPProblem(
        N=4, dt=0.25,
        x0=np.array([1.0]), u0=0.0,
        dyn=dyn, m_sub=2,
        nu=1, nx=1,
        objective_builder=kwargs.pop("objective_builder",
                                     make_builders(dyn=dyn, L=L, Phi=None)),
        control_bounds_builder=lambda p: [(-1.0, 1.0)] * (p.N * p.nu),
        **kwargs,
    )


def test_a_numpy_object_array_gets_the_jax_hint():
    """The most common leftover from the CppAD days: np.array(..., dtype=object)."""
    def dyn(x, u, theta=None):
        return np.array([u[0] - x[0]], dtype=object)

    with pytest.raises(TypeError, match="jax.numpy"):
        OCPSolver(_problem(dyn)).to_nlp()


def test_numpy_math_functions_get_the_jax_hint():
    """np.exp cannot see inside a tracer; jnp.exp can."""
    def dyn(x, u, theta=None):
        return jnp.array([np.exp(-x[0]) + u[0]])

    with pytest.raises(TypeError, match="jnp.exp"):
        OCPSolver(_problem(dyn)).to_nlp()


def test_an_if_on_a_value_gets_the_jax_hint():
    """
    A Python if/else on a state would silently freeze one branch into an AD tape;
    under JAX it must fail loudly instead, pointing at jnp.where.
    """
    def dyn(x, u, theta=None):
        return jnp.array([u[0] if x[0] > 0.0 else -u[0]])

    with pytest.raises(TypeError, match="jnp.where"):
        OCPSolver(_problem(dyn)).to_nlp()


def test_the_jnp_where_version_of_the_same_model_works():
    """...and the fix the hint suggests really does work."""
    def dyn(x, u, theta=None):
        return jnp.array([jnp.where(x[0] > 0.0, u[0], -u[0])])

    nlp = OCPSolver(_problem(dyn)).to_nlp()
    assert np.isfinite(nlp.objective_fun(np.full(4, 0.5)))


def test_dynamics_of_the_wrong_length_are_named():
    def dyn(x, u, theta=None):
        return jnp.array([u[0], 0.0])          # two entries for a one-state problem

    with pytest.raises(ValueError, match="dyn must return an array of shape"):
        OCPSolver(_problem(dyn)).to_nlp()


def test_a_different_dyn_in_make_builders_is_refused():
    """
    There is exactly one simulation, driven by OCPProblem(dyn=...). A different
    function given to make_builders would be ignored without a word, so it is
    refused instead.
    """
    def dyn(x, u, theta=None):
        return jnp.array([u[0]])

    def other_dyn(x, u, theta=None):
        return jnp.array([2.0 * u[0]])

    p = _problem(dyn, objective_builder=make_builders(dyn=other_dyn, L=None, Phi=None))
    with pytest.raises(ValueError, match="same dynamics"):
        OCPSolver(p).to_nlp()


def test_a_running_cost_must_be_a_single_number():
    def dyn(x, u, theta=None):
        return jnp.array([u[0]])

    def L(t, x, u, theta):
        return jnp.array([x[0], u[0]])         # a vector where a number is expected

    with pytest.raises(ValueError, match="L must return a single number"):
        OCPSolver(_problem(dyn, L=L)).to_nlp()
