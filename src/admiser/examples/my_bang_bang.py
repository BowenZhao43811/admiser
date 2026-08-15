# my_bang_bang.py -- canonical-constraint version (this problem has no constraints)

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # builds the objective_builder only

# --- grid / dimensions ---
T, N = 1.0, 21
dt = T / N
nx, nu = 2, 2

# initial state
x0 = np.array([1.0, 0.0], dtype=float)

# control initial guess
u0 = np.array([1.0, 1.0], dtype=float)

# --- dynamics ---
# x = [x1, x2], u = [u1, u2]
def dyn(x, u, theta=None):
    x1, x2 = x
    u1, u2 = u
    dx1 = u2
    dx2 = -x1 + u1
    return np.array([dx1, dx2], dtype=object)

# --- integrand L(t, x, u) ---
# J = int_0^1 ( -6*x1 - 12*x2 + 3*u1 + u2 ) dt
def L(t, x, u, theta):
    return (-6.0 * x[0]) + (-12.0 * x[1]) + (3.0 * u[0]) + (u[1])

# Only the objective builder is needed; constraints would go through the
# canonical add_* API (this example has none).
objective_builder = make_builders(
    dyn=dyn,
    L=L,
    Phi=None,
    quad='rk4',
)

# --- control box bounds ---
def control_bounds_builder(problem: OCPProblem):
    return [(-10.0, 10.0)] * (problem.N * problem.nu)

# --- assemble the problem ---
problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,                # no xT; no terminal, path or integral constraints here
    u0 = u0,
    dyn=dyn,
    integrator=partial(rk4_substeps, m_sub=10),
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0,             # no system parameters
)

# 4th-order quadrature, matching the state integrator (see QUAD_SCHEMES)
problem.quad_scheme = 'rk4'

__all__ = ["problem", "N", "dt", "T"]
