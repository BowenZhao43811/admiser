# my_bang_bang.py  —— 新版（canonical constraints，无显式约束）

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # 仅构建 objective_builder

# --- 网格 / 维度 ---
T, N = 1.0, 21
dt = T / N
nx, nu = 2, 2

# 初始状态
x0 = np.array([1.0, 0.0], dtype=float)

# 控制初猜
u0 = np.array([1.0, 1.0], dtype=float)

# --- 动力学 ---
# x = [x1, x2], u = [u1, u2]
def dyn(x, u, theta=None):
    x1, x2 = x
    u1, u2 = u
    dx1 = u2
    dx2 = -x1 + u1
    return np.array([dx1, dx2], dtype=object)

# --- 被积函数 L(t,x,u) ---
# J = ∫_0^1 ( -6 x1 - 12 x2 + 3 u1 + u2 ) dt
def L(t, x, u, theta):
    return (-6.0 * x[0]) + (-12.0 * x[1]) + (3.0 * u[0]) + (u[1])

# 只需要目标构建器；约束全部走 canonical API（本例没有约束）
objective_builder = make_builders(
    dyn=dyn,
    L=L,
    Phi=None,
    quad='rk4',
)

# --- 控制盒约束 ---
def control_bounds_builder(problem: OCPProblem):
    return [(-10.0, 10.0)] * (problem.N * problem.nu)

# --- 组装问题 ---
problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,                # 无 xT；本例无终端/路径/积分约束
    u0 = u0,
    dyn=dyn,
    integrator=partial(rk4_substeps, m_sub=10),
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0,             # 无系统参数
)

# 采样位置（子步中点）
problem.quad_scheme = 'rk4'

__all__ = ["problem", "N", "dt"]
