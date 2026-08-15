# my_port_kobe.py -- six-state container-crane problem, canonical-constraint version

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # builds int L dt + Phi only

# ========== grid ==========
T  = 1.0
N  = 20                   # start at 20; refine to 200/300/400 once it converges
dt = T / N

nx, nu = 6, 2

# ========== initial and terminal states ==========
x0 = np.array([0.0, 22.0, 0.0, 0.0, -1.0, 0.0], dtype=float)
xT = np.array([10.0, 14.0, 0.0, 2.5, 0.0, 0.0], dtype=float)

# ========== dynamics ==========
def dyn(x, u, theta=None):
    """
    dx1/dt = 9*x4
    dx2/dt = 9*x5
    dx3/dt = 9*x6
    dx4/dt = 9*(u1 + 17.2656*x3)
    dx5/dt = 9*u2
    dx6/dt = -9/x2 * (u1 + 27.0756*x3 + 2*x5*x6)
    """
    x1, x2, x3, x4, x5, x6 = x
    u1, u2 = u

    dx1 = 9.0 * x4
    dx2 = 9.0 * x5
    dx3 = 9.0 * x6
    dx4 = 9.0 * (u1 + 17.2656 * x3)
    dx5 = 9.0 * u2
    dx6 = -9.0 / x2 * (u1 + 27.0756 * x3 + 2.0 * x5 * x6)

    return np.array([dx1, dx2, dx3, dx4, dx5, dx6], dtype=object)

# ========== objective: 4.5 * int_0^1 [x3^2 + x6^2] dt ==========
def L(t, x, u, theta):
    return 4.5 * (x[2]*x[2] + x[5]*x[5])

# objective only (Phi = None)
objective_builder= make_builders(
    dyn=dyn,
    L=L,
    Phi=None,
    quad='rk4',
)

# ========== control box bounds ==========
# |u1| <= 2.83374;  -0.80865 <= u2 <= 0.71265
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((-2.83374,  2.83374))   # u1
        b.append((-0.80865,  0.71265))   # u2
    return b

# ========== assemble the problem ==========
substepped_rk4 = partial(rk4_substeps, m_sub=12)  # more substeps for stability

# ---- terminal equality x(T) - xT = 0, via the canonical API ----
def terminal_eq_psi(xT_ad, atheta):
    xT_const = np.array([v for v in xT], dtype=object)
    return xT_ad - xT_const

# ========== path inequalities (constraint transcription) ==========
# h1 = x4 - 2.5 <= 0
# h2 = -x4 - 2.5 <= 0
# h3 = x5 - 1.0 <= 0
# h4 = -x5 - 1.0 <= 0
def h1(t_ad, x_ad, u_ad, atheta):  # x4 <= 2.5
    return x_ad[3] - 2.5

def h2(t_ad, x_ad, u_ad, atheta):  # x4 >= -2.5
    return -x_ad[3] - 2.5

def h3(t_ad, x_ad, u_ad, atheta):  # x5 <= 1.0
    return x_ad[4] - 1.0

def h4(t_ad, x_ad, u_ad, atheta):  # x5 >= -1.0
    return -x_ad[4] - 1.0

PATH_INEQS = (h1, h2, h3, h4)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    dyn=dyn, integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0, theta0=None, param_bounds_builder=None,
)
problem.quad_scheme = 'rk4'
problem.add_terminal_eq(terminal_eq_psi)

# All four path constraints share one eps. Omitting gamma gives the consistent
# value gamma = T*eps/4 automatically, kept in sync as eps shrinks.
for _h in PATH_INEQS:
    problem.add_path_ineq(_h, eps=1e-2)

# ===== solve mode: declared here, the solving side just calls OCPSolver(problem).solve() =====
# eps is the FINAL value; the rounds run 1e0 -> 1e-1 -> 1e-2, warm-started,
# with all four constraints shrinking together.
problem.set_transcription(mode="continuation", n_rounds=3, shrink=0.1)

__all__ = ["problem", "T", "N", "dt", "PATH_INEQS"]
