# my_min_x4_terminal_heading.py
import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # 仅构建 objective_builder

# ===== 网格 =====
T  = 1.0
N  = 10
dt = T / N

nx, nu = 4, 2

# ===== 初始状态 =====
x0 = np.array([np.pi/2, 4.0, 0.0, 0.0], dtype=float)  # [x1, x2, x3, x4]

u0 = [1.0, 1.0]  # 可选初猜

# ===== 动力学 =====
# dx1 = u2 * u1
# dx2 = u2 * cos(x1)
# dx3 = u2 * sin(x1)
# dx4 = u2
def dyn(x, u, theta=None):
    x1, x2, x3, x4 = x
    u1, u2 = u
    dx1 = u2 * u1
    dx2 = u2 * np.cos(x1)
    dx3 = u2 * np.sin(x1)
    dx4 = u2
    return np.array([dx1, dx2, dx3, dx4], dtype=object)

# ===== 目标：min x4(T) =====
L = None

def Phi(xT_ad, theta):
    return xT_ad[3]  # x4(T)

objective_builder= make_builders(
    dyn=dyn,
    L=L,         # None/无效返回 => 仅终端代价
    Phi=Phi,
    quad='rk4',
)

# ===== 控制盒约束 =====
# u1 ∈ [-2, 2]；u2 不设边界（SLSQP可用 (None, None) 表示无界）
def control_bounds_builder(problem: OCPProblem):
    bounds = []
    for _ in range(problem.N):
        bounds.append((-2.0,  2.0))     # u1_k
        bounds.append((0.0,  100.0))    # u2_k
    return bounds

# ===== 组装问题 =====
substepped_rk4 = partial(rk4_substeps, m_sub=10)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0, u0 = u0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0,
)
problem.path_quad_mode = 'rk4'

# 终端等式：x2(T)=0, x3(T)=0
def terminal_eq_psi(xT_ad, theta):
    # 返回等式向量 = 0
    return np.array([
        xT_ad[1] - 0.0,  # x2(T) = 0
        xT_ad[2] - 0.0,  # x3(T) = 0
    ], dtype=object)

problem.add_terminal_eq(terminal_eq_psi)

__all__ = ["problem", "N", "dt", "T"]
