# my_ratio_control_problem.py
import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders # 仅构建 objective_builder

# ===== 网格 =====
T  = 1.0
N  = 10
dt = T / N

nx, nu = 3, 3

# ===== 初始状态 =====
x0 = np.array([0.8, 0.7, 0.0], dtype=float)  # [x1, x2, x3]
u0 = [0.01, 0.01, 0.01]  # 可选初猜

# ===== 动力学 =====
# dx1/dt = ((1-x1)u1 + (2-x1)u2) * u3 / x2
# dx2/dt = (-0.02*sqrt(x2) + u1 + u2) * u3
# dx3/dt = x3
def dyn(x, u, theta=None):
    x1, x2, x3 = x
    u1, u2, u3 = u

    dx1 = ((1.0 - x1) * u1 + (2.0 - x1) * u2) * u3 / x2
    dx2 = (-0.02 * np.sqrt(x2) + u1 + u2) * u3
    dx3 = u3

    return np.array([dx1, dx2, dx3], dtype=object)

# ===== 目标：min x3(T)（无积分项）=====
L = None
def Phi(xT_ad, theta):
    return xT_ad[2]  # x3(T)

# 兼容 make_builders 返回 1 或 2 个对象
_ret = make_builders(dyn=dyn, L=L, Phi=Phi, quad='rk4-mid')
objective_builder = _ret[0] if isinstance(_ret, tuple) else _ret

# ===== 控制盒约束 =====
# u1 ∈ [0, 0.03]；u2 ∈ [0, 0.01]；u3 ≥ 0 （无上界）
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((0.0, 0.03))  # u1_k
        b.append((0.0, 0.01))  # u2_k
        b.append((0.0, 100.0))  # u3_k
    return b

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
problem.path_quad_mode = 'rk4-mid'

# 终端等式：x1(T)=1.25, x2(T)=1.0 （x3 终端自由）
def terminal_eq_psi(xT_ad, theta):
    return np.array([
        xT_ad[0] - 1.25,
        xT_ad[1] - 1.0,
    ], dtype=object)

problem.add_terminal_eq(terminal_eq_psi)

# —— 可选：若担心 sqrt 域或分母过小，可启用路径不等式以稳定计算 ——
# def h_x2_floor(t, x, u, th):  # 要求 x2 >= 1e-6  <=>  1e-6 - x2 ≤ 0
#     return cppad_py.a_double(1e-6) - x[1]
# problem.add_path_ineq(hfun=h_x2_floor, eps=1e-4, gamma=0.0)

__all__ = ["problem", "N", "dt", "T"]
