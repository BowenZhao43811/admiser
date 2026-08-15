# my_robot_problem.py -- canonical-constraint version

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders   # builds the objective only

# ---------------- basic setup ----------------
T  = 12.0
N  = 40
dt = T / N

# initial and target terminal states
x0 = np.array([-10.0, -10.0, np.pi/2, 0.0, 0.0, 0.0], dtype=float)
xT = np.zeros(6, dtype=float)

nu, nx = 2, 6

u0 = [1.0, 1.0]

# System parameter, a fixed constant here. To optimise it instead, read it from
# atheta[0] in dyn (see the commented block further down).
alpha = 0.2

# ---------------- dynamics ----------------
# x = [x1, x2, x3, x4, x5, x6], u = [u1, u2]
def dyn(x, u, atheta=None):
    # For the parameter-optimisation variant, replace this with: a = atheta[0]
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

# ---------------- objective L(t, x, u) ----------------
# J = int (u1^2 + u2^2) dt
def L(t, x, u, atheta):
    return u[0]*u[0] + u[1]*u[1]

# Only the objective builder is needed; constraints go through the add_* API.
objective_builder= make_builders(
    dyn=dyn,
    L=L,
    Phi=None,
    quad='rk4',   # 4th-order quadrature, matching the state (see QUAD_SCHEMES)
)

# ---------------- control box bounds ----------------
u1_max, u2_max = 0.8, 0.4
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((-u1_max, u1_max))  # u1_k
        b.append((-u2_max, u2_max))  # u2_k
    return b

# ------- optional: treat alpha as a system parameter to be optimised -------
# def param_bounds_builder(problem: OCPProblem):
#     return [(0.05, 1.0)]  # alpha in [0.05, 1.0]
# theta0 = np.array([0.25])

# ---------------- assemble the problem ----------------
substepped_rk4 = partial(rk4_substeps, m_sub=10)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,                 # fixed x0; if it depends on theta use x0_from_theta_ad/_numeric
    u0 = u0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0,              # to optimise alpha: set to 1 and pass theta0/param_bounds_builder
    # theta0=theta0,
    # param_bounds_builder=param_bounds_builder,
)
problem.quad_scheme = 'rk4'

# ---- terminal equality x(T) - xT = 0, registered through the canonical API ----
def terminal_eq_psi(xT_ad, atheta):
    xT_const = np.array([v for v in xT], dtype=object)
    return xT_ad - xT_const   # vector equality = 0

problem.add_terminal_eq(terminal_eq_psi)

__all__ = ["problem", "N", "dt", "T", "u1_max", "u2_max"]
