# my_free_terminal_time2.py -- ratio control problem, minimum-time via an extra state
import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # builds the objective_builder only

# ===== grid =====
T  = 1.0
N  = 10
dt = T / N

nx, nu = 3, 3

# ===== initial state =====
x0 = np.array([0.8, 0.7, 0.0], dtype=float)  # [x1, x2, x3]
u0 = [0.01, 0.01, 0.01]  # optional initial guess

# ===== dynamics =====
# dx1/dt = ((1-x1)*u1 + (2-x1)*u2) * u3 / x2
# dx2/dt = (-0.02*sqrt(x2) + u1 + u2) * u3
# dx3/dt = u3
def dyn(x, u, theta=None):
    x1, x2, x3 = x
    u1, u2, u3 = u

    dx1 = ((1.0 - x1) * u1 + (2.0 - x1) * u2) * u3 / x2
    dx2 = (-0.02 * np.sqrt(x2) + u1 + u2) * u3
    dx3 = u3

    return np.array([dx1, dx2, dx3], dtype=object)

# ===== objective: min x3(T), no running cost =====
L = None
def Phi(xT_ad, theta):
    return xT_ad[2]  # x3(T)

objective_builder = make_builders(dyn=dyn, L=L, Phi=Phi, quad='rk4')

# ===== control box bounds =====
# u1 in [0, 0.03];  u2 in [0, 0.01];  u3 in [0, 100]
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((0.0, 0.03))   # u1_k
        b.append((0.0, 0.01))   # u2_k
        b.append((0.0, 100.0))  # u3_k
    return b

# ===== assemble the problem =====
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
problem.quad_scheme = 'rk4'

# terminal equalities: x1(T) = 1.25, x2(T) = 1.0 (x3(T) is free)
def terminal_eq_psi(xT_ad, theta):
    return np.array([
        xT_ad[0] - 1.25,
        xT_ad[1] - 1.0,
    ], dtype=object)

problem.add_terminal_eq(terminal_eq_psi)

# Optional: if the sqrt argument or the denominator get too small, a path
# inequality can keep x2 away from zero.
# def h_x2_floor(t, x, u, th):   # require x2 >= 1e-6, i.e. 1e-6 - x2 <= 0
#     return 1e-6 - x[1]
# problem.add_path_ineq(hfun=h_x2_floor, eps=1e-4)

__all__ = ["problem", "N", "dt", "T"]
