# my_state_control_constraint.py -- a mixed state-and-control path constraint

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # builds int L dt only

# ===== grid =====
T  = 4.5
N  = 20
dt = T / N

nx, nu = 2, 1

# ===== initial state =====
x0 = np.array([-5.0, -5.0], dtype=float)

u0 = 1.0  # initial guess: constant control

# ===== dynamics (3-argument signature; theta is unused here) =====
def dyn(x, u, theta):
    """
    dx1/dt = x2
    dx2/dt = -x1 + x2*(1.4 - 0.14*x2^2) + 4*u
    """
    x1, x2 = x
    uu     = u[0]              # a_double or float
    dx1 = x2
    dx2 = -x1 + x2 * (1.4 - 0.14* x2*x2) + 4 * uu
    return np.array([dx1, dx2], dtype=object)

# ===== objective L: int (x1^2 + u^2) dt =====
def L(t, x, u, theta):
    x1, x2 = x
    uu = u[0]
    return x1*x1 + uu*uu

# no terminal cost, no terminal or integral equalities in this example
Phi = None

# ===== objective_builder only; constraints go through the add_* API =====
objective_builder= make_builders(
    dyn=dyn,
    L=L,
    Phi=Phi,
    quad='rk4',   # 4th-order quadrature, matching the state (see QUAD_SCHEMES)
)

# ===== control box bounds =====
def control_bounds_builder(problem: OCPProblem):
    # one u_k per segment, in [-20, 20]
    return [(-20.0, 20.0)] * (problem.N * problem.nu)

# ===== assemble the problem (multi-substep RK4 for stability) =====
substepped_rk4 = partial(rk4_substeps, m_sub=20)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    u0 = u0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0, theta0=None, param_bounds_builder=None,
)
problem.quad_scheme = 'rk4'

# ===== mixed state-control path inequality: h(t, x, u) = u + x1/6 <= 0 =====
def h(t_ad, x_ad, u_ad, atheta):
    x1, x2 = x_ad
    uu = u_ad[0]
    return uu + 1/6 * x1                   # feasible where h <= 0

# Teo's transcription: int L_eps(h) dt <= gamma.
# Omitting gamma gives gamma = T*eps/4 automatically, kept in sync as eps shrinks.
problem.add_path_ineq(h, eps=1e-3)

# ===== solve mode: declared here, the solving side just calls OCPSolver(problem).solve() =====
# eps is the FINAL value; the rounds run 1e0 -> 1e-1 -> 1e-2 -> 1e-3, warm-started.
problem.set_transcription(mode="continuation", n_rounds=4, shrink=0.1)

__all__ = ["problem", "N", "dt", "T", "h"]
