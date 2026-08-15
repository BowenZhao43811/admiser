# my_ocp_problem_tpl.py
# 通用 OCP 问题定义模板（新接口：目标与约束解耦；约束统一 canonical 注册）
import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders   # 仅构建 objective_builder

# ============= 1) 网格与维度 =============
T  = 1.0
N  = 100
dt = T / N
nx = 2
nu = 1

# ============= 2) 初值/（可选）目标终端值 =============
x0 = np.zeros(nx, dtype=float)
u0 = None  # 控制初猜；若无特殊要求可设为 None
# 若需要 x(T)=xT 的等式约束，会在下面用 problem.add_terminal_eq(...) 注册
xT = None

# ============= 3) 动力学（必须支持 theta 可为 None） =============
def dyn(x, u, theta=None):
    """
    示例：
      x = [x1, x2], u = [u1]
      x1' = x2
      x2' = -x1 + u1
    """
    x1, x2 = x
    u1 = u[0]
    dx1 = x2
    dx2 = -x1 + u1
    return np.array([dx1, dx2], dtype=object)

# ============= 4) 目标函数片段（L 与 Phi） =============
def L(t, x, u, theta):
    """示例：L = x1^2 + x2^2 + 1e-3 * u^2"""
    return x[0]*x[0] + x[1]*x[1] + 1e-3 * (u[0]*u[0])

def Phi(xT_ad, theta):
    """无终端代价则返回 None；示例保留 None"""
    return None

# 仅构建“目标” builder；约束统一用 add_* 注册
objective_builder = make_builders(
    dyn=dyn,
    L=L,           # 可为 None
    Phi=Phi,       # 可为 None
    quad='rk4' # 子步求积模式
)

# ============= 5) 可选：初值依赖 θ（x0 = x0(θ)） =============
def x0_from_theta_ad(atheta, problem):
    """
    示例：x0 = [θ1, 0]；若无参数优化，可不传
    """
    if atheta is None:
        return np.array([0.0, 0.0], dtype=object)
    return np.array([atheta[0], 0.0], dtype=object)

def x0_from_theta_numeric(theta, problem):
    if theta is None:
        return x0.copy()
    return np.array([float(theta[0]), 0.0], dtype=float)

# ============= 6) 控制盒约束 =============
def control_bounds_builder(problem: OCPProblem):
    """示例：|u1| <= 10"""
    return [(-10.0, 10.0)] * (problem.N * problem.nu)

# ============= 7) 系统参数 θ（可选） =============
ntheta = 0
theta0 = None

def param_bounds_builder(problem: OCPProblem):
    """若 ntheta>0，返回与 θ 维度匹配的盒约束列表。示例：θ1 ∈ [0, 1]"""
    return [(0.0, 1.0)]

# ============= 8) 组装 Problem =============
substepped_rk4 = partial(rk4_substeps, m_sub=10)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,                         # 若 x0 依赖 θ，会在 AD/数值路径中被覆盖
    u0=u0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=ntheta,
    theta0=theta0,
    param_bounds_builder=param_bounds_builder if ntheta > 0 else None,
    x0_from_theta_ad=x0_from_theta_ad if ntheta > 0 else None,
    x0_from_theta_numeric=x0_from_theta_numeric if ntheta > 0 else None,
)
problem.path_quad_mode = 'rk4'

# ============= 9) 约束注册 =============
# ⚠️ 下面六类是一份"菜单"，彼此**不是**同时可行的（例如 9.1 要求 x1(T)=1，
#    而 9.6 要求 x1(T)≤0.5）。本模板默认只启用 9.1 + 9.4 + 9.5 这一组自洽的
#    约束，其余以注释形式保留写法。按你自己的问题取用，别一次全打开。

# 9.1 终端等式：x(T) = xT
def terminal_eq_psi(xT_ad, atheta):
    xT_const = np.array([v for v in [1.0, 0.0]], dtype=object)
    return xT_ad - xT_const   # 向量等式=0
problem.add_terminal_eq(terminal_eq_psi)

# 9.2 积分等式：∫ q(t,x,u,θ) dt = target
# def q_int(t, x, u, th): return x[0]
# problem.add_integral_eq(qfun=q_int, target=1.0)

# 9.3 路径等式：h(t,x,u,θ) = 0
# - L2 模式：∫ h^2 dt = 0（推荐）
# - abs 模式：∫ smooth_abs_eps(h) dt = 0，eps_abs 会被 solve_transcription 一并收缩
# def h_eq(t, x, u, th): return x[1]
# problem.add_path_eq(hfun=h_eq, mode="L2", eps_abs=1e-6)

# 9.4 积分不等式：∫ q dt ≤/≥ bound
def q_budget(t, x, u, th): return u[0]*u[0]   # 示例：控制能量预算
problem.add_integral_ineq(qfun=q_budget, bound=50.0, sense="<=")

# 9.5 路径不等式（约束转译）：h(t,x,u,θ) ≤ 0  →  ∫ L_eps(h) dt ≤ γ
# gamma 省略即按 γ = T·ε/4 自动取值（与 ε 相容的取法，见 OCPProblem.auto_gamma）。
# 切勿写 gamma=0.0：那等价于强制 h(t) ≤ -ε（严格内点），既保守又难收敛。
# ε 是**有量纲**的，要按 h 自身的量级取；不确定时用 solve_transcription 从大到小续贯。
def h_ineq(t, x, u, th): return x[1] - 3.0   # 示例：|x2| 不超过 3
problem.add_path_ineq(hfun=h_ineq, eps=1e-2)

# 9.6 终端不等式：φ(xT,θ) ≤ 0 或 ≥ 0
# def phi_T(xT_ad, th): return xT_ad[0] - 0.5
# problem.add_terminal_ineq(phi=phi_T, sense="<=")

# ============= 10) 模板导出 =============
__all__ = ["problem", "N", "dt"]
