# my_bitumen_pyrolysis.py -- bitumen pyrolysis, temperature as the control
# A general OCP template: multi-substep RK4 + CppAD_py AD + optional system
# parameters theta + constraint transcription, all registered through one API.

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # builds int L dt + Phi only

# ================== 1) grid and dimensions ==================
T  = 10.0          # total physical duration (fixed-T case)
N  = 30
dt = T / N
nx = 2
nu = 1

# ================== 2) initial state / optional terminal target ==================
x0 = np.array([1.0, 0.0], dtype=float)

u0 = np.array([723.15], dtype=float)  # optional guess: constant temperature, 450 C
# For a terminal equality x(T) = xT, register it with problem.add_terminal_eq(...) below.
# xT = np.array([...], dtype=float)

# ================== 3) dynamics ==================
# Must have the signature dyn(x, u, theta=None)
def dyn(x, u, theta=None):
    """
    x = [x1, x2], u = [u1] with u1 the absolute temperature in Kelvin.
    Five Arrhenius rate constants drive the reaction network.
    """
    x1, x2 = x
    u1 = u[0]
    ln_a = np.array([8.86, 24.25, 23.67, 18.75, 20.70], dtype=float)
    b_over_R = np.array([10215.4, 18820.5, 17008.9, 14190.8, 15599.8], dtype=float)
    k = np.exp(-b_over_R / u1 + ln_a)
    k1, k2, k3, k4, k5 = k
    dx1 = -k1 *x1 - (k3 + k4 + k5) * x1 * x2
    dx2 = k1 *x1 - k2 * x2 + k3 * x1 * x2
    return np.array([dx1, dx2], dtype=object)

# ================== 4) objective pieces (L and/or Phi) ==================
# This example uses a terminal cost only: maximising x2(T) is minimising -x2(T).
def Phi(xT_ad, theta):
    return -xT_ad[1]

# For a running cost, define something like:
# def L(t, x, u, theta):
#     return 0.01 * (u[0]*u[0])  # e.g. an energy term
L = None

# Assemble the objective_builder (no constraints are attached here any more)
objective_builder= make_builders(
    dyn=dyn,
    L=L,         # may be None
    Phi=Phi,     # may be None
    quad='rk4',
)

# ================== 5) control box bounds ==================
def control_bounds_builder(problem: OCPProblem):
    # temperature limits
    return [(698.15, 748.15)] * (problem.N * problem.nu)

# ================== 6) optional: system parameters theta, and x0(theta) ==================
# For system parameters, set ntheta, theta0 and param_bounds_builder, and use
# atheta inside dyn. If the initial state depends on theta, supply
# x0_from_theta_ad / x0_from_theta_numeric.

# ================== 7) assemble the problem ==================
substepped_rk4 = partial(rk4_substeps, m_sub=10)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    u0 = u0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0,              # set > 0 for system parameters, plus theta0/param_bounds_builder
    # theta0=...,
    # param_bounds_builder=...,
)
problem.quad_scheme = 'rk4'

# ================== 8) register constraints (canonical API) ==================
# Terminal equality, if x(T) = xT is wanted:
# def psi_terminal(xT_ad, theta):
#     xT_const = np.array([v for v in xT], dtype=object)
#     return xT_ad - xT_const
# problem.add_terminal_eq(psi_terminal)

# Integral equality: int q(t, x, u, theta) dt = target
# problem.add_integral_eq(lambda t,x,u,th: u[0], target=123.0)

# Integral inequality: int q dt <= / >= bound
# problem.add_integral_ineq(lambda t,x,u,th: u[0], bound=200.0, sense="<=")

# Path inequality (constraint transcription): h(x,u,theta) <= 0 -> int L_eps(h) dt <= gamma
# Omit gamma to get the consistent value T*eps/4 automatically.
# problem.add_path_ineq(lambda t,x,u,th: x[1] - 0.8, eps=1e-3)

# Path equality (transcribed into an integral equality): h(t)=0 -> int h^2 dt = 0
# problem.add_path_eq(lambda t,x,u,th: x[1], mode="L2")

# ================== 9) exports ==================
__all__ = ["problem", "N", "dt", "T"]
