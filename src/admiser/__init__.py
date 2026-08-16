"""
ADMISER -- optimal control by control parametrization, with automatic
differentiation in place of the adjoint equations.

The public surface is deliberately small. Almost every problem needs only:

    from admiser import OCPProblem, OCPSolver, make_builders, rk4_substeps

Everything else lives in the modules below and can be imported from there when
needed. Each public name is a promise to keep, so this list stays short on purpose.

    problem_definition     OCPProblem: the problem and how it should be solved
    objective_builder      make_builders: assembles int L dt + Phi
    constraint_smoothing   L_eps, smooth_abs: the smoothing behind the transcription
    quadrature             the substep integrator and the quadrature scheme family
    ad_tape                records the whole problem onto one AD tape
    nlp_functions          presents that tape to SciPy, with caching and scaling
    problem_scaling        the automatic objective/constraint scaling
    sqp_solver             OCPSolver: the SLSQP driver and continuation loop
"""

from ._version import __version__

_CPPAD_HINT = (
    "ADMISER needs cppad_py to record its automatic-differentiation tapes, "
    "but it was not found in this environment.\n"
    "Run the following inside the ACTIVATED virtual environment that has admiser installed:\n"
    "    admiser-install-cppad\n"
    "That command builds and installs CppAD and cppad_py "
    "(tested on Debian/Ubuntu/WSL only)."
)

try:
    import cppad_py as _cppad_py  # noqa: F401
except ImportError as _exc:       # pragma: no cover - depends on the user's environment
    raise ImportError(f"{_exc}\n\n{_CPPAD_HINT}") from _exc

# ---- the everyday API ----
from .problem_definition import OCPProblem
from .sqp_solver import OCPSolver
from .objective_builder import make_builders
from .quadrature import rk4_substeps, QUAD_SCHEMES

# ---- helpers a user genuinely writes into their own model functions ----
# These are AD-safe by construction: they record CppAD conditional expressions
# rather than resolving a Python branch once, at taping time.
from .constraint_smoothing import L_eps, smooth_abs

__all__ = [
    "__version__",
    "OCPProblem",
    "OCPSolver",
    "make_builders",
    "rk4_substeps",
    "QUAD_SCHEMES",
    "L_eps",
    "smooth_abs",
]
