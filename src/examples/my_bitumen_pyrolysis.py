# my_ocp_problem_tpl.py  —— 新版（统一约束接口）
# 通用 OCP 模板：多子步RK4 + CppAD_py AD + 可扩展参数θ + 约束转译（统一注册）

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # 仅构建 ∫L + Φ

# ================== 1) 网格与维度 ==================
T  = 10.0          # 物理总时长（固定T场景用于参考；若做CPET/自由T，dt_k由求解器定义）
N  = 30
dt = T / N
nx = 2
nu = 1

# ================== 2) 初值/（可选）目标终端 ==================
x0 = np.array([1.0, 0.0], dtype=float)

u0 = np.array([723.15], dtype=float)  # 可选初猜：恒定温度 450°C
# 如需终端等式 x(T)=xT，请在文末用 problem.add_terminal_eq(...) 注册
# xT = np.array([...], dtype=float)

# ================== 3) 动力学 ==================
# 必须满足签名 dyn(x, u, theta=None)
def dyn(x, u, theta=None):
    """
    示例（改成你的表达式）:
        x = [x1, x2], u = [u1]
        x1' = x2
        x2' = -x1 + u1
    """
    x1, x2 = x
    u1 = u[0]
    ln_a = np.array([8.86, 24.25, 23.67, 18.75, 20.70], dtype=float)
    b_over_R = np.array([10215.4, 18820.5, 17008.9, 14190.8, 15599.8], dtype=float)
    k = np.exp(-b_over_R / u1 + ln_a)  # 假设 x2 是温度，单位 °C
    k1, k2, k3, k4, k5 = k
    dx1 = -k1 *x1 - (k3 + k4 + k5) * x1 * x2
    dx2 = k1 *x1 - k2 * x2 + k3 * x1 * x2
    return np.array([dx1, dx2], dtype=object)

# ================== 4) 目标函数片段（L 与/或 Φ） ==================
# 这里示例仅用终端代价：最大化 x1(T) 等价于最小化 -x1(T)
def Phi(xT_ad, theta):
    return -xT_ad[1]

# 若需要运行阶段代价，可定义：
# def L(t, x, u, theta):
#     return 0.01 * (u[0]*u[0])  # 例如能耗项
L = None

# 用模板拼装 objective_builder（不再从这里挂任何约束）
objective_builder= make_builders(
    dyn=dyn,
    L=L,         # 可为 None
    Phi=Phi,     # 可为 None
    quad='rk4-mid',
)

# ================== 5) 控制盒约束 ==================
def control_bounds_builder(problem: OCPProblem):
    # 温度上下界（示例）
    return [(698.15, 748.15)] * (problem.N * problem.nu)

# ================== 6) （可选）系统参数 θ & 初值随 θ ==================
# 若有系统参数，请设置 ntheta、theta0、param_bounds_builder，并在 dyn 中使用 atheta。
# 若初值依赖 θ，可提供 x0_from_theta_ad / x0_from_theta_numeric。

# ================== 7) 组装 Problem ==================
substepped_rk4 = partial(rk4_substeps, m_sub=10)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    u0 = u0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0,              # ← 有系统参数时改为 >0，并补充 theta0/param_bounds_builder
    # theta0=...,
    # param_bounds_builder=...,
)
problem.path_quad_mode = 'rk4-mid'

# ================== 8) 在此统一注册约束（canonical constraints） ==================
# 终端等式（如果需要 x(T)=xT）：
# def psi_terminal(xT_ad, theta):
#     xT_const = np.array([cppad_py.a_double(v) for v in xT], dtype=object)
#     return xT_ad - xT_const
# problem.add_terminal_eq(psi_terminal)

# 积分等式：∫ q(t,x,u,θ) dt = target
# problem.add_integral_eq(lambda t,x,u,th: u[0], target=123.0)

# 积分不等式：∫ q dt ≤ / ≥ bound
# problem.add_integral_ineq(lambda t,x,u,th: u[0], bound=200.0, sense="<=")

# 路径不等式（约束转译）：h(x,u,θ) ≤ 0 -> ∫ L_eps(h) dt ≤ γ
# from smooth_utils import L_eps  # 若想用在自定义 qfun 里
# problem.add_path_ineq(lambda t,x,u,th: x[1] - 0.8, eps=1e-6, gamma=0.0)

# 路径等式（转译为积分等式）：h(t)=0 -> ∫ h^2 dt = 0
# problem.add_path_eq(lambda t,x,u,th: x[1], mode="L2")

# ================== 9) （可选）启用 CPET/自由终端时间 ==================
# 若以后接入 time-scaling（CPET），这里会提供 _enable_cpet(problem) 之类的开关
# _enable_cpet(problem)

# 对外暴露
__all__ = ["problem", "N", "dt"]
