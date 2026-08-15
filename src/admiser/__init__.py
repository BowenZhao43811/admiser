from ._version import __version__

_CPPAD_HINT = (
    "ADMISER 需要 cppad_py 才能构建自动微分带（AD tape），但当前环境里没有找到它。\n"
    "请在**已激活**的、装有 admiser 的虚拟环境中运行：\n"
    "    admiser-install-cppad\n"
    "该命令会编译并安装 CppAD 与 cppad_py（仅在 Debian/Ubuntu/WSL 上测试通过）。"
)

try:
    import cppad_py as _cppad_py  # noqa: F401
except ImportError as _exc:       # pragma: no cover - 取决于用户环境
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

__all__ = [
    "__version__",
    "OCPProblem", "OCPSolver", "make_builders",
    "rk4_substeps", "rk4_step",
    "QuadScheme", "QUAD_SCHEMES", "DEFAULT_QUAD_SCHEME",
    "quad_order", "validate_quad_scheme",
    "L_eps", "smooth_abs",
    "ensure_cppad_or_warn",
]


# a method for checking cppad_py installation
def ensure_cppad_or_warn() -> bool:
    """返回 cppad_py 是否可用；不可用时打印安装提示（而不是抛异常）。"""
    try:
        import cppad_py  # noqa: F401
    except ImportError:
        print(f"[ADMISER] WARNING: {_CPPAD_HINT}")
        return False
    return True
