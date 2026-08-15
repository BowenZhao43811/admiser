# my_medical1.py -- multi-compartment tumour therapy, canonical-constraint version

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # builds the objective_builder only

# ================= 1) grid and dimensions =================
T  = 30.0
N  = 120
dt = T / N

nx = 6  # [S1, S2, S3, N, C1, C2]
nu = 2  # [u, v]

# ================= 2) model parameters =================
# growth rates and carrying capacities
r1 = 0.25; r2 = 0.15; r3 = 0.15; r4 = 1.2
K1 = 1.0e6; K2 = 1.0e6; K3 = 1.0e6; K4 = 1.0e6

# migration / transition rates
tau1 = 0.001; tau2 = 0.001
tau23 = 0.0001; tau32 = 0.0001

# drug effect coefficients on S1/S2/S3 and on N
a1 = 0.45; a2 = 0.45; a3 = 0.45
b1 = 0.15
aN = 0.02; bN = 0.02

# extra growth term for N: coefficient gamma_V and threshold T_star
gamma_V = 0.028
T_star  = 3.0e5

# pharmacokinetic decay
lam1 = 0.01
lam2 = 0.01

# stage-cost weights in the objective
q1 = 1.0; q2 = 1.0; q3 = 1.0
beta1 = 1; beta2 = 3

# dose budget: int (u + v) dt <= budget_c
budget_c = 20.0

# control box bounds
u_min, u_max = 0.0, 1.0
v_min, v_max = 0.0, 1.0

# initial state
x0 = np.array([7e5, 1e5, 1e5, 1e6, 1.0, 1.0], dtype=float)  # [S1, S2, S3, N, C1, C2]

# ================= 3) dynamics =================
def dyn(x, u, theta=None):
    """
    state:   x = [S1, S2, S3, N, C1, C2]
    control: u = [u, v]
    """
    S1, S2, S3, N, C1, C2 = x
    u1, v1 = u   # u, v in [0, 1]
    V = S1 + S2 + S3

    # S1
    dS1 = (r1 * S1 * (1.0 - V / K1)
           - tau1 * S1 - tau2 * S1
           - a1 * (1.0 - np.exp(-C1)) * S1
           - b1 * (1.0 - np.exp(-C2)) * S1)

    # S2
    dS2 = (r2 * S2 * (1.0 - V / K2)
           + tau1 * S1
           - a2 * (1.0 - np.exp(-C2)) * S2
           - tau23 * (1.0 - np.exp(-C2)) * S2
           + tau32 * (1.0 - np.exp(-C1)) * S3)

    # S3
    dS3 = (r3 * S3 * (1.0 - V / K3)
           + tau2 * S1
           - a3 * (1.0 - np.exp(-C1)) * S3
           - tau32 * (1.0 - np.exp(-C1)) * S3
           + tau23 * (1.0 - np.exp(-C2)) * S2)

    # N
    dN  = (r4 * N * (1.0 - N / K4)
           + gamma_V * V * (1.0 - V / T_star)
           - aN * (1.0 - np.exp(-C1)) * N
           - bN * (1.0 - np.exp(-C2)) * N)

    # drug concentrations
    dC1 = u1 - lam1 * C1
    dC2 = v1 - lam2 * C2

    return np.array([dS1, dS2, dS3, dN, dC1, dC2], dtype=object)

# ================= 4) objective =================
def L(t, x, u, theta):
    """q1*S1 + q2*S2 + q3*S3 + beta1*u + beta2*v"""
    return (q1*x[0]) + (q2*x[1]) + (q3*x[2]) + (beta1*u[0]) + (beta2*u[1])

Phi = None  # no terminal cost

objective_builder= make_builders(
    dyn=dyn,
    L=L,
    Phi=Phi,
    quad='midpoint',
)

# ================= 5) control box bounds =================
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((u_min, u_max))  # u
        b.append((v_min, v_max))  # v
    return b

# ================= 6) assemble the problem =================
substepped_rk4 = partial(rk4_substeps, m_sub=10)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0, theta0=None, param_bounds_builder=None,
)
problem.quad_scheme = 'midpoint'  # 2nd-order quadrature, preserving this example's original numbers

# ================= 7) register the constraints (canonical API) =================
# 7.1 path inequality: N(t) >= 0.75*K4, i.e. h = 0.75*K4 - N(t) <= 0
def h_N_floor(t, x, u, theta):
    return 0.75 * K4 - x[3]  # x[3] = N
eps0, gamma0 = 1e-3, 1e-8
problem.add_path_ineq(hfun=h_N_floor, eps=eps0, gamma=gamma0)

# 7.2 dose budget: int (u + v) dt <= budget_c
def q_budget(t, x, u, theta):
    return u[0] + u[1]
problem.add_integral_ineq(qfun=q_budget, bound=budget_c, sense="<=")

# ================= 8) exports =================
__all__ = ["problem", "N", "dt", "T", "K1","K2","K3","K4","budget_c"]
