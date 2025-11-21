# my_covid_seir_ocp.py
# COVID-19 SEIR 最优控制：u1=疫苗接种率, u2=接触率抑制强度
import numpy as np, cppad_py
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders

# ===== 时间网格 =====
T  = 90.0                # 天
N  = 30                 # 每天1步
dt = T / N

# ===== SEIR 参数（单位：1/天；可按需调整）=====
beta   = 0.6              # 基础传播率
sigma  = 1.0 / 5.2        # 潜伏→感染 转移率
gamma  = 1.0 / 10.0       # 康复率
rho    = 0.9              # 接触率抑制有效系数：beta_eff = beta * (1 - rho * u2)

# ===== 初始状态（人口总量归一化）=====
I0 = 0.005
E0 = 0.010
R0 = 0.0
S0 = 1.0 - (E0 + I0 + R0)
x0 = np.array([S0, E0, I0, R0], dtype=float)   # [S,E,I,R]
nx, nu = 4, 2

# ===== 动力学 =====
# u = [u1, u2]：u1=疫苗接种率（S->R），u2=接触率抑制强度（降低beta）
def dyn(x, u, theta=None):
    S, E, I, R = x
    u1, u2 = u
    beta_eff = beta * (1.0 - rho * u2)

    dS = - beta_eff * S * I - u1 * S
    dE =   beta_eff * S * I - sigma * E
    dI =   sigma * E - gamma * I
    dR =   gamma * I + u1 * S
    return np.array([dS, dE, dI, dR], dtype=object)

# ===== 目标：∫ (wI*I + wE*E + a1*u1^2 + a2*u2^2) dt =====
wI, wE = 50.0, 5.0
a1, a2 = 1.0, 10.0
def L(t, x, u, theta):
    S, E, I, R = x
    u1, u2 = u
    return (wI*I) + (wE*E) + (a1*(u1*u1)) + (a2*(u2*u2))

Phi = None  # 无终端代价

# 由模板生成 objective_builder（内部在多子步 RK4 中精确积分 ∫L·dt）
_ret = make_builders(
    dyn=dyn, L=L, Phi=Phi,
    quad='rk4-mid',    # 子步采样为中点，数值更稳
)
objective_builder = _ret if callable(_ret) else _ret[0]
constraint_builder = None  # 约束统一用 canonical API 注册

# ===== 盒约束 =====
u1_max = 0.20     # 每天最大接种分率
u2_max = 0.80     # 最大接触率抑制强度
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((0.0, u1_max))  # u1
        b.append((0.0, u2_max))  # u2
    return b

# ===== 构造 Problem（多子步 RK4）=====
substepped_rk4 = partial(rk4_substeps, m_sub=10)
problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    u0=0.0,                      # 初猜控制全 0（可改成 0.5 等）
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    constraint_builder=constraint_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0, theta0=None, param_bounds_builder=None,
)
problem.path_quad_mode = 'rk4-mid'

# ====== 注册约束（canonical API）======
# 1) 路径不等式：I(t) ≤ I_cap  →  h = I - I_cap ≤ 0 → ∫ L_eps(h) dt ≤ γ
I_cap = 0.02
def h_peak_I(t, x, u, theta):
    return x[2] - cppad_py.a_double(I_cap)   # x[2]=I
problem.add_path_ineq(h_peak_I, eps=1e-2, gamma=0.25e-2)  # γ 非零更利于数值

# 2) 积分不等式：累计接种剂量 ≤ 预算
#   更合理的剂量是 ∫(u1*S)dt（每单位时间接种给 S），而不是 ∫u1 dt
V_budget = 0.60
def q_vax(t, x, u, theta):   # 每步消耗
    S, E, I, R = x
    u1 = u[0]
    return u1 * S
problem.add_integral_ineq(qfun=q_vax, bound=V_budget, sense="<=")

__all__ = ["problem", "N", "dt", "T", "u1_max", "u2_max", "I_cap", "V_budget"]


