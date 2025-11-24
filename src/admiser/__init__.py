from ._version import __version__

# Directly exposed common APIs
from .ocp_problem_builders import OCPProblem
from .ocp_solver_builders import OCPSolver
from .ocp_integrators_utils import rk4_substeps
from .ocp_function_builders import make_builders

# a method for checking cppad_py installation
def ensure_cppad_or_warn():
    try:
        import cppad_py
    except Exception:
        print("[ADMISER] WARNING: cppad_py not found. "
              "Please run tools/build_cppad_py.sh to build & install it for your environment.")
