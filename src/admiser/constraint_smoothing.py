# constraint_smoothing.py
"""
The two smooth functions behind the constraint transcription.

Both are written with jax.numpy, and the kink of L_eps is handled with jnp.where,
never with a Python if/else on the value. Inside the solver every quantity is a
JAX "tracer": a stand-in that records the operations applied to it, so they can
be compiled and differentiated afterwards. A Python `if` needs a concrete
True/False, which a tracer does not have, so JAX raises an error rather than
guess. jnp.where keeps BOTH branches in the computation and picks between them
elementwise, so the value and the derivative are right on either side of a kink.

Both functions also accept plain floats and NumPy arrays, so they can be called
directly to evaluate or plot a constraint.
"""

import jax.numpy as jnp


def smooth_abs(h, eps_abs):
    """Smooth approximation of |h|: sqrt(h^2 + eps_abs^2)."""
    return jnp.sqrt(h * h + eps_abs * eps_abs)


def L_eps(h, eps):
    """
    Smoothed hinge L_eps(h) ~ max(h, 0), in three pieces:

      h <= -eps  -> 0
      h >= +eps  -> h
      otherwise  -> (h + eps)^2 / (4*eps)

    The pieces meet with equal values AND equal slopes at h = -eps and h = +eps,
    so L_eps is continuously differentiable, which is what a gradient-based
    optimiser needs. Note that in the middle band L_eps is positive even where
    the constraint h <= 0 holds; that is why the transcribed constraint allows a
    small tolerance gamma (see OCPProblem.auto_gamma).
    """
    middle = (h + eps) * (h + eps) / (4.0 * eps)
    return jnp.where(h <= -eps, 0.0, jnp.where(h >= eps, h, middle))
