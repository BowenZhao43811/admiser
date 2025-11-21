from ._version import __version__

# 常用API直接暴露
from .ocp_problem_builders import OCPProblem
from .ocp_solver_builders import OCPSolver
from .ocp_integrators_utils import rk4_substeps
from .ocp_function_builders import make_builders

# 检测 cppad_py 是否正确安装
from .ad_check import ensure_cppad_or_warn as _ensure_cppad
_ensure_cppad()
