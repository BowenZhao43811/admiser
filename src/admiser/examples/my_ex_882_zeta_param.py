import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders

# ---- 网格 ----
T = 1.0
N = 100
dt = T / N
nu, nx = 1, 2

u0 = 1.0  # 初猜：恒定控制

# ---- 常数 ----
beta  = 0.5
p_pow = 2
gamma = 0.1
u_max = 6.0

# ---- x0(θ) = [(1-γ)+γζ1, γζ2] ----
def x0_from_theta_ad(atheta, problem):
    z1, z2 = atheta[0], atheta[1]
    return np.array([
        1.0 - gamma + gamma * z1,
        gamma * z2
    ], dtype=object)

def x0_from_theta_numeric(theta, problem):
    z1, z2 = float(theta[0]), float(theta[1])
    return np.array([(1.0 - gamma) + gamma*z1, gamma*z2], dtype=float)

# ---- 动力学 ----
def dyn(x, u, _theta_unused):
    x1, x2 = x[0], x[1]
    uu = u[0]
    dx1 = -(uu + beta * (uu**p_pow)) * x1
    dx2 =  uu * x1
    return np.array([dx1, dx2], dtype=object)

# ---- 目标：min −ζ2 （用终端代价 Phi 来实现）----
def Phi(xT_ad, atheta):
    z2 = atheta[1]
    return -z2

# 目标 builder（只构建 ∫L + Φ；这里 L=None, 仅终端代价）
objective_builder= make_builders(dyn=dyn, L=None, Phi=Phi, quad='rk4-mid')

# ---- 盒约束 ----
def control_bounds_builder(problem: OCPProblem):
    return [(0.0, u_max)] * (problem.N * problem.nu)

# ---- 参数 θ = [ζ1, ζ2] 与其盒约束 ----
ntheta = 2
theta0 = np.array([0.8, 0.3], dtype=float)
def param_bounds_builder(problem: OCPProblem):
    return [(0.0, 1.0), (0.0, 1.0)]

# ---- 组装 ----
substep_rk4 = partial(rk4_substeps, m_sub=10)
problem = OCPProblem(
    N=N, dt=dt,
    x0=np.zeros(nx),  # 占位；真实初值由 θ 决定
    u0 = u0,
    dyn=dyn,
    integrator=substep_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=ntheta, theta0=theta0, param_bounds_builder=param_bounds_builder,
    x0_from_theta_ad=x0_from_theta_ad,
    x0_from_theta_numeric=x0_from_theta_numeric,
)
problem.path_quad_mode = 'rk4-mid'

# 终端等式：x(T) = θ
def terminal_eq_psi(xT_ad, atheta):
    return xT_ad - atheta  # 2维向量等式
problem.add_terminal_eq(terminal_eq_psi)

__all__ = ["problem", "N", "dt", "u_max"]
