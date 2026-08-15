# my_multi_compartment_therapy_tpl.py  —— 新版（canonical constraints）

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # 只构建 objective_builder

# ================= 1) 网格与维度 =================
T  = 30.0
N  = 120
dt = T / N

nx = 6  # [S1,S2,S3,N,C1,C2]
nu = 2  # [u, v]

# ================= 2) 模型参数（占位，可按需调整） ================
# 增长/环境容量
r1 = 0.25; r2 = 0.15; r3 = 0.15; r4 = 1.2
K1 = 1.0e6; K2 = 1.0e6; K3 = 1.0e6; K4 = 1.0e6

# 迁移/转移率
tau1 = 0.001; tau2 = 0.001
tau23 = 0.0001; tau32 = 0.0001

# 药效（对 S1/S2/S3 与 N 的效应系数）
a1 = 0.45; a2 = 0.45; a3 = 0.45
b1 = 0.15
aN = 0.02; bN = 0.02

# N 的额外增长项参数 γ 与 T*
gamma_V = 0.028
T_star  = 3.0e5

# 药物动力学衰减
lam1 = 0.01
lam2 = 0.01

# 阶段成本权重（目标函数）
q1 = 1.0; q2 = 1.0; q3 = 1.0
beta1 = 1; beta2 = 3

# 预算上限（∫ (u+v) dt ≤ c）
budget_c = 20.0

# 控制盒约束
u_min, u_max = 0.0, 1.0
v_min, v_max = 0.0, 1.0

# 初始状态（可按需改）
x0 = np.array([7e5, 1e5, 1e5, 1e6, 1.0, 1.0], dtype=float)  # [S1,S2,S3,N,C1,C2]

# ================= 4) 动力学 =================
def dyn(x, u, theta=None):
    """
    状态: x = [S1,S2,S3,N,C1,C2]
    控制: u = [u, v]
    """
    S1, S2, S3, N, C1, C2 = x
    u1, v1 = u   # u,v ∈ [0,1]
    V = S1 + S2 + S3

    # S1
    dS1 = (r1 * S1 * (1.0 - V / K1)
           - tau1 * S1 - tau2 * S1
           - a1 * (1.0 - np.exp(-C1)) * S1
           - b1 * (1.0 - np.exp(-C2)) * S1)

    # S2
    dS2 = (r2 * S2 * (1.0 - V / K2)
           + tau1 * S1
           - a2 * (1.0 - np.exp(-C2)) * S2
           - tau23 * (1.0 - np.exp(-C2)) * S2
           + tau32 * (1.0 - np.exp(-C1)) * S3)

    # S3
    dS3 = (r3 * S3 * (1.0 - V / K3)
           + tau2 * S1
           - a3 * (1.0 - np.exp(-C1)) * S3
           - tau32 * (1.0 - np.exp(-C1)) * S3
           + tau23 * (1.0 - np.exp(-C2)) * S2)

    # N
    dN  = (r4 * N * (1.0 - N / K4)
           + gamma_V * V * (1.0 - V / T_star)
           - aN * (1.0 - np.exp(-C1)) * N
           - bN * (1.0 - np.exp(-C2)) * N)

    # 药物浓度
    dC1 = u1 - lam1 * C1
    dC2 = v1 - lam2 * C2

    return np.array([dS1, dS2, dS3, dN, dC1, dC2], dtype=object)

# ================= 5) 目标函数 =================
def L(t, x, u, theta):
    """q₁S₁ + q₂S₂ + q₃S₃ + β₁u + β₂v"""
    return (q1*x[0]) + (q2*x[1]) + (q3*x[2]) + (beta1*u[0]) + (beta2*u[1])

Phi = None  # 无终端代价

# 只构建目标（约束统一走 add_* API）
objective_builder= make_builders(
    dyn=dyn,
    L=L,
    Phi=Phi,
    quad='midpoint',
)

# ================= 6) 控制盒约束 =================
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((u_min, u_max))  # u
        b.append((v_min, v_max))  # v
    return b

# ================= 7) 构造 Problem =================
substepped_rk4 = partial(rk4_substeps, m_sub=10)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0, theta0=None, param_bounds_builder=None,
)
problem.quad_scheme = 'midpoint'  # 2 阶求积，保持本例原有数值

# ================= 8) 约束注册（canonical） =================
# 8.1 路径不等式（转译）：N(t) - 0.75 K4 ≥ 0  <=>  0.75*K4 - N(t) ≤ 0
def h_N_floor(t, x, u, theta):
    return 0.75 * K4 - x[3]  # x[3] = N
eps0, gamma0 = 1e-3, 1e-8
problem.add_path_ineq(hfun=h_N_floor, eps=eps0, gamma=gamma0)

# 8.2 预算：∫ (u+v) dt ≤ budget_c
def q_budget(t, x, u, theta):
    return u[0] + u[1]
problem.add_integral_ineq(qfun=q_budget, bound=budget_c, sense="<=")

# ================= 9) 对外暴露 =================
__all__ = ["problem", "N", "dt", "T", "K1","K2","K3","K4","budget_c"]
