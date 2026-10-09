"""
ADMISER -- optimal control by control parametrization, with automatic
differentiation in place of the adjoint equations.

The public surface is deliberately small. Almost every problem needs only:

    from admiser import OCPProblem, OCPSolver, make_builders

Everything else lives in the modules below and can be imported from there when
needed. Each public name is a promise to keep, so this list stays short on purpose.

    problem_definition     OCPProblem: the problem and how it should be solved
    objective_builder      make_builders: declares int L dt + Phi
    constraint_smoothing   L_eps, smooth_abs: the smoothing behind the transcription
    quadrature             the RK4 substep and the quadrature scheme family
    ocp_to_nlp             turns the problem into compiled NLP functions (JAX)
    nlp_functions          presents those functions to SciPy, with caching and scaling
    problem_scaling        the automatic objective/constraint scaling
    sqp_solver             OCPSolver: the SLSQP driver and continuation loop
"""

from ._version import __version__

import jax as _jax

# JAX computes in 32-bit floating point unless told otherwise. That is far too
# coarse here: SLSQP compares objective changes against ftol ~ 1e-8 and beyond,
# and float32 carries only about 7 significant digits. So admiser switches JAX to
# 64-bit as soon as it is imported. It must happen before any JAX array is
# created, which is why it sits at the very top of the package.
#
# The setting is process-wide: other JAX code running in the same Python session
# will compute in 64-bit as well.
_jax.config.update("jax_enable_x64", True)

# ---- the everyday API ----
from .problem_definition import OCPProblem
from .sqp_solver import OCPSolver
from .objective_builder import make_builders
from .quadrature import QUAD_SCHEMES

# ---- helpers a user genuinely writes into their own model functions ----
# Written with jnp.where, so they stay correct and differentiable on both sides of
# their kink when JAX traces them.
from .constraint_smoothing import L_eps, smooth_abs

__all__ = [
    "__version__",
    "OCPProblem",
    "OCPSolver",
    "make_builders",
    "QUAD_SCHEMES",
    "L_eps",
    "smooth_abs",
]
