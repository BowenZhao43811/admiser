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

# Directly exposed common APIs
from .ocp_problem_builders import OCPProblem, DEFAULT_QUAD_SCHEME
from .ocp_solver_builders import OCPSolver
from .ocp_integrators_utils import (
    rk4_substeps, rk4_step,
    QuadScheme, QUAD_SCHEMES, quad_order, validate_quad_scheme,
)
from .ocp_function_builders import make_builders
from .ocp_smooth_utils import L_eps, smooth_abs
from .ocp_scaling_utils import ProblemScaling, compute_scaling, identity_scaling

__all__ = [
    "__version__",
    "OCPProblem", "OCPSolver", "make_builders",
    "rk4_substeps", "rk4_step",
    "QuadScheme", "QUAD_SCHEMES", "DEFAULT_QUAD_SCHEME",
    "quad_order", "validate_quad_scheme",
    "L_eps", "smooth_abs",
    "ProblemScaling", "compute_scaling", "identity_scaling",
    "ensure_cppad_or_warn",
]


def ensure_cppad_or_warn() -> bool:
    """
    Report whether cppad_py is importable. Prints the installation hint instead of
    raising when it is not.
    """
    try:
        import cppad_py  # noqa: F401
    except ImportError:
        print(f"[ADMISER] WARNING: {_CPPAD_HINT}")
        return False
    return True
