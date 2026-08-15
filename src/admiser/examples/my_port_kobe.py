# my_port_kobe_tpl.py  —— 新版（canonical constraints）

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # 仅构建 ∫L+Φ 的 objective_builder

# ========== 网格 ==========
T  = 1.0
N  = 20                   # 可先 20；收敛后再加密：200/300/400
dt = T / N

nx, nu = 6, 2

# ========== 初末状态 ==========
x0 = np.array([0.0, 22.0, 0.0, 0.0, -1.0, 0.0], dtype=float)
xT = np.array([10.0, 14.0, 0.0, 2.5, 0.0, 0.0], dtype=float)

# ========== 动力学 ==========
def dyn(x, u, theta=None):
    """
    dx1/dt = 9*x4
    dx2/dt = 9*x5
    dx3/dt = 9*x6
    dx4/dt = 9*(u1 + 17.2656*x3)
    dx5/dt = 9*u2
    dx6/dt = -9 * x2 * (u1 + 27.0756*x3 + 2*x5*x6)   # ← 按题设：乘 x2
    """
    x1, x2, x3, x4, x5, x6 = x
    u1, u2 = u

    dx1 = 9.0 * x4
    dx2 = 9.0 * x5
    dx3 = 9.0 * x6
    dx4 = 9.0 * (u1 + 17.2656 * x3)
    dx5 = 9.0 * u2
    dx6 = -9.0 / x2 * (u1 + 27.0756 * x3 + 2.0 * x5 * x6)

    return np.array([dx1, dx2, dx3, dx4, dx5, dx6], dtype=object)

# ========== 目标： 4.5 * ∫_0^1 [(x3)^2 + (x6)^2] dt ==========
def L(t, x, u, theta):
    return 4.5 * (x[2]*x[2] + x[5]*x[5])

# 仅构建目标（Φ=None）
objective_builder= make_builders(
    dyn=dyn,
    L=L,
    Phi=None,
    quad='rk4',
)

# ========== 控制盒约束 ==========
# |u1| ≤ 2.83374； -0.80865 ≤ u2 ≤ 0.71265
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((-2.83374,  2.83374))   # u1
        b.append((-0.80865,  0.71265))   # u2
    return b

# ========== 构造问题 ==========
substepped_rk4 = partial(rk4_substeps, m_sub=12)  # 子步多些更稳

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    dyn=dyn, integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0, theta0=None, param_bounds_builder=None,
)
problem.path_quad_mode = "rk4"

# ---- 终端等式：x(T) - xT = 0（用 canonical API 注册）----
def terminal_eq_psi(xT_ad, atheta):
    xT_const = np.array([v for v in xT], dtype=object)
    return xT_ad - xT_const

problem.add_terminal_eq(terminal_eq_psi)

# ========== 路径不等式（约束转译） ==========
# h1 = x4 - 2.5 ≤ 0
# h2 = -x4 - 2.5 ≤ 0
# h3 = x5 - 1.0 ≤ 0
# h4 = -x5 - 1.0 ≤ 0
def h1(t_ad, x_ad, u_ad, atheta):  # x4 <= 2.5
    return x_ad[3] - 2.5

def h2(t_ad, x_ad, u_ad, atheta):  # x4 >= -2.5
    return -x_ad[3] - 2.5

def h3(t_ad, x_ad, u_ad, atheta):  # x5 <= 1.0
    return x_ad[4] - 1.0

def h4(t_ad, x_ad, u_ad, atheta):  # x5 >= -1.0
    return -x_ad[4] - 1.0

# 平滑宽度 eps 与 CQ 余量 gamma（可做同伦：逐轮 eps *= 0.5）
eps0   = 1e-2
gamma0 = 1e-3
problem.add_path_ineq(h1, eps=eps0, gamma=gamma0)
problem.add_path_ineq(h2, eps=eps0, gamma=gamma0)
problem.add_path_ineq(h3, eps=eps0, gamma=gamma0)
problem.add_path_ineq(h4, eps=eps0, gamma=gamma0)

__all__ = ["problem", "T", "N", "dt"]
