# my_ex_882_zeta_param.py -- system parameters theta = [zeta1, zeta2] that also set x0
import numpy as np
import jax.numpy as jnp

from admiser import OCPProblem
from admiser import make_builders

# ---- grid ----
T = 1.0
N = 100
dt = T / N
nu, nx = 1, 2

u0 = 1.0  # initial guess: constant control

# ---- constants ----
beta  = 0.5
p_pow = 2
gamma = 0.1
u_max = 6.0

# ---- x0(theta) = [(1-gamma) + gamma*z1, gamma*z2] ----
def x0_from_theta(theta, problem):
    z1, z2 = theta[0], theta[1]
    return jnp.array([
        1.0 - gamma + gamma * z1,
        gamma * z2
    ])

# ---- dynamics ----
def dyn(x, u, _theta_unused):
    x1, x2 = x[0], x[1]
    uu = u[0]
    dx1 = -(uu + beta * (uu**p_pow)) * x1
    dx2 =  uu * x1
    return jnp.array([dx1, dx2])

# ---- objective: min -zeta2, expressed as a terminal cost Phi ----
def Phi(xT_ad, atheta):
    z2 = atheta[1]
    return -z2

# Objective builder only (int L dt + Phi); here L is None, so terminal cost only.
objective_builder= make_builders(dyn=dyn, L=None, Phi=Phi, quad='rk4')

# ---- control box bounds ----
def control_bounds_builder(problem: OCPProblem):
    return [(0.0, u_max)] * (problem.N * problem.nu)

# ---- system parameters theta = [zeta1, zeta2] and their box bounds ----
ntheta = 2
theta0 = np.array([0.8, 0.3], dtype=float)
def param_bounds_builder(problem: OCPProblem):
    return [(0.0, 1.0), (0.0, 1.0)]

# ---- assemble ----
problem = OCPProblem(
    N=N, dt=dt,
    x0=np.zeros(nx),  # placeholder; the real initial state comes from theta
    u0 = u0,
    dyn=dyn,
    m_sub=10,         # RK4 substeps per segment
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=ntheta, theta0=theta0, param_bounds_builder=param_bounds_builder,
    x0_from_theta=x0_from_theta,
)
problem.quad_scheme = 'rk4'

# terminal equality: x(T) = theta
def terminal_eq_psi(xT_ad, atheta):
    return xT_ad - atheta  # 2-dimensional vector equality
problem.add_terminal_eq(terminal_eq_psi)

__all__ = ["problem", "N", "dt", "T", "u_max"]
