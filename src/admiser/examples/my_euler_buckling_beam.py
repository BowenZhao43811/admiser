import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders

# 网格
T = 1.0
N = 21
dt = T / N
nu, nx = 1, 3

u0 = 1.0  # 初猜：恒定控制

# 动力学
# x1' = x2
# x2' = -z1 * x1 / x3^2
# x3' = u
def dyn(x, u, atheta):
    z1, z2 = atheta[0], atheta[1]
    x1, x2, x3 = x
    uu = u[0]
    dx1 = x2
    dx2 = -(z1 * x1) / (x3 * x3)
    dx3 = uu
    return np.array([dx1, dx2, dx3], dtype=object)

# 目标：min -z1  -> 用 Phi
def Phi(xT_ad, atheta):
    return -atheta[0]

objective_builder= make_builders(dyn=dyn, L=None, Phi=Phi, quad='rk4')

# 初值依赖 θ：x1(0)=0, x2(0)=1, x3(0)=z2
def x0_from_theta_ad(atheta, problem):
    return np.array([0.0,
                     1.0,
                     atheta[1]], dtype=object)
def x0_from_theta_numeric(theta, problem):
    return np.array([0.0, 1.0, float(theta[1])], dtype=float)

# 盒约束（可选）：限制 u 以帮助数值稳定
def control_bounds_builder(problem: OCPProblem):
    return [(-10.0, 10.0)] * (problem.N * problem.nu)

# 参数与盒界：z1 ≥ 0、z2 ≥ 0.5
ntheta = 2
theta0 = np.array([5.0, 0.6], dtype=float)
def param_bounds_builder(problem: OCPProblem):
    return [(0.0, 50.0), (0.5, 5.0)]

substep_rk4 = partial(rk4_substeps, m_sub=10)
problem = OCPProblem(
    N=N, dt=dt,
    x0=np.zeros(nx), dyn=dyn, integrator=substep_rk4,
    nu=nu, nx=nx, u0 = u0,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=ntheta, theta0=theta0, param_bounds_builder=param_bounds_builder,
    x0_from_theta_ad=x0_from_theta_ad,
    x0_from_theta_numeric=x0_from_theta_numeric,
)
problem.path_quad_mode = 'rk4'

# 终端等式：x1(1)=0
problem.add_terminal_eq(lambda xT, th: xT[0])

# 积分等式：∫ x3 dt = 1
problem.add_integral_eq(qfun=lambda t,x,u,th: x[2], target=1.0)

# 路径不等式：x3(t) ≥ 0.5  →  h = 0.5 - x3 ≤ 0  → add_path_ineq(h, eps, γ=0)
problem.add_path_ineq(hfun=lambda t,x,u,th: 0.5 - x[2],
                      eps=1e-6, gamma=1e-8)

__all__ = ["problem", "N", "dt"]
