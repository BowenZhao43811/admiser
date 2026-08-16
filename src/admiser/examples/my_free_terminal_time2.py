# my_free_terminal_time2.py -- minimum-time ratio control, via the time-scaling transform
#
# Reach a target concentration and volume in the least possible time.
#
# The earlier formulation carried a third control u3 that multiplied every right
# hand side, plus a third state x3 accumulating it, and minimised x3(T). That is
# the time-scaling transform written out by hand. It is now built in: the
# dynamics are stated in real time, the durations tau_k are decision variables,
# and the objective is the elapsed time.

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders

# ===== grid =====
# N segments; dt is only the initial guess for each duration, the horizon is free.
N  = 10
dt = 10.0

nx, nu = 2, 2

# ===== initial state =====
x0 = np.array([0.8, 0.7], dtype=float)   # [concentration, volume]
u0 = [0.01, 0.01]                        # initial guess for the two feed rates

# ===== dynamics, in real time =====
# dx1/dt = ((1 - x1)*u1 + (2 - x1)*u2) / x2
# dx2/dt =  -0.02*sqrt(x2) + u1 + u2
def dyn(x, u, theta=None):
    x1, x2 = x
    u1, u2 = u
    dx1 = ((1.0 - x1) * u1 + (2.0 - x1) * u2) / x2
    dx2 = -0.02 * np.sqrt(x2) + u1 + u2
    return np.array([dx1, dx2], dtype=object)

# ===== objective: minimise the elapsed time =====
def L(t, x, u, theta):
    return 1.0

objective_builder = make_builders(dyn=dyn, L=L, Phi=None, quad='rk4')

# ===== control box bounds =====
# u1 in [0, 0.03];  u2 in [0, 0.01]
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((0.0, 0.03))   # u1_k
        b.append((0.0, 0.01))   # u2_k
    return b

# ===== assemble the problem =====
problem = OCPProblem(
    N=N, dt=dt,
    x0=x0, u0=u0,
    dyn=dyn,
    integrator=partial(rk4_substeps, m_sub=10),
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0,
)
problem.quad_scheme = 'rk4'

# ===== the time-scaling transform =====
# x2 sits under a square root and in a denominator, so a strictly positive lower
# bound on the durations keeps the very first iterations away from degenerate steps.
problem.set_time_scaling(tau0=dt, tau_min=1e-3, tau_max=200.0)

# ===== terminal equalities: x1(T) = 1.25, x2(T) = 1.0 =====
def terminal_eq_psi(xT_ad, theta):
    return np.array([
        xT_ad[0] - 1.25,
        xT_ad[1] - 1.0,
    ], dtype=object)

problem.add_terminal_eq(terminal_eq_psi)

__all__ = ["problem", "N", "dt"]
