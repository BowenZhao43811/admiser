# my_robot_problem_tpl.py  —— 新版（canonical constraints）

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders   # ← 新接口：仅构建目标

# ---------------- 基本设置 ----------------
T  = 12.0
N  = 40
dt = T / N

# 初末
x0 = np.array([-10.0, -10.0, np.pi/2, 0.0, 0.0, 0.0], dtype=float)
xT = np.zeros(6, dtype=float)

nu, nx = 2, 6.

u0 = [1.0, 1.0]

# 系统参数（缺省常数；如要做“参数优化”，把 a 改成来自 atheta[0]）
alpha = 0.2

# ---------------- 动力学 ----------------
# x = [x1,x2,x3,x4,x5,x6], u = [u1,u2]
def dyn(x, u, atheta=None):
    # 如需“参数优化版本”，替换为：a = atheta[0]
    a = alpha

    x1, x2, x3, x4, x5, x6 = x
    u1, u2 = u
    s = u1 + u2
    dx1 = x4
    dx2 = x5
    dx3 = x6
    dx4 = s * np.cos(x3)
    dx5 = s * np.sin(x3)
    dx6 = a * (u1 - u2)
    return np.array([dx1, dx2, dx3, dx4, dx5, dx6], dtype=object)

# ---------------- 目标 L(t,x,u) ----------------
# J = ∫ (u1^2 + u2^2) dt
def L(t, x, u, atheta):
    return u[0]*u[0] + u[1]*u[1]

# 用模板生成：只需要 objective_builder；约束统一走 add_* API
objective_builder= make_builders(
    dyn=dyn,
    L=L,
    Phi=None,
    quad='rk4-mid',   # 你的 integrator 已支持子步采样位置
)

# ---------------- 盒约束 ----------------
u1_max, u2_max = 0.8, 0.4
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((-u1_max, u1_max))  # u1_k
        b.append((-u2_max, u2_max))  # u2_k
    return b

# ------- （可选）把 alpha 当成系统参数来优化 -------
# def param_bounds_builder(problem: OCPProblem):
#     return [(0.05, 1.0)]  # alpha ∈ [0.05, 1.0]
# theta0 = np.array([0.25])

# ---------------- 组装问题 ----------------
substepped_rk4 = partial(rk4_substeps, m_sub=10)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,                 # 固定初值；若初值依赖 theta，可用 x0_from_theta_ad/ numeric
    u0 = u0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0,              # ← 若优化 alpha：改为 1，并传 theta0/param_bounds_builder
    # theta0=theta0,
    # param_bounds_builder=param_bounds_builder,
)
problem.path_quad_mode = 'rk4-mid'

# ---- 终端等式：x(T) - xT = 0  （用 canonical API 注册）----
def terminal_eq_psi(xT_ad, atheta):
    xT_const = np.array([v for v in xT], dtype=object)
    return xT_ad - xT_const   # 向量等式 = 0

problem.add_terminal_eq(terminal_eq_psi)

__all__ = ["problem", "N", "dt", "u1_max", "u2_max"]
