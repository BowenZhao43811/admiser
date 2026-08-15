# my_free_terminal_time.py -- minimise x4(T) subject to a terminal heading condition
import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # builds the objective_builder only

# ===== grid =====
T  = 1.0
N  = 10
dt = T / N

nx, nu = 4, 2

# ===== initial state =====
x0 = np.array([np.pi/2, 4.0, 0.0, 0.0], dtype=float)  # [x1, x2, x3, x4]

u0 = [1.0, 1.0]  # optional initial guess

# ===== dynamics =====
# dx1 = u2 * u1
# dx2 = u2 * cos(x1)
# dx3 = u2 * sin(x1)
# dx4 = u2
def dyn(x, u, theta=None):
    x1, x2, x3, x4 = x
    u1, u2 = u
    dx1 = u2 * u1
    dx2 = u2 * np.cos(x1)
    dx3 = u2 * np.sin(x1)
    dx4 = u2
    return np.array([dx1, dx2, dx3, dx4], dtype=object)

# ===== objective: min x4(T) =====
L = None

def Phi(xT_ad, theta):
    return xT_ad[3]  # x4(T)

objective_builder= make_builders(
    dyn=dyn,
    L=L,         # None => terminal cost only
    Phi=Phi,
    quad='rk4',
)

# ===== control box bounds =====
# u1 in [-2, 2];  u2 in [0, 100]
def control_bounds_builder(problem: OCPProblem):
    bounds = []
    for _ in range(problem.N):
        bounds.append((-2.0,  2.0))     # u1_k
        bounds.append((0.0,  100.0))    # u2_k
    return bounds

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

# terminal equalities: x2(T) = 0, x3(T) = 0
def terminal_eq_psi(xT_ad, theta):
    # the returned vector must equal zero
    return np.array([
        xT_ad[1] - 0.0,  # x2(T) = 0
        xT_ad[2] - 0.0,  # x3(T) = 0
    ], dtype=object)

problem.add_terminal_eq(terminal_eq_psi)

__all__ = ["problem", "N", "dt", "T"]
