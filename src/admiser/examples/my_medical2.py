# my_medical2.py -- COVID-19 SEIR optimal control
# u1 = vaccination rate, u2 = contact-reduction intensity
import numpy as np, cppad_py
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders

# ===== time grid =====
T  = 90.0               # days
N  = 30
dt = T / N

# ===== SEIR parameters (units 1/day) =====
beta   = 0.6              # baseline transmission rate
sigma  = 1.0 / 5.2        # exposed -> infected transition rate
gamma  = 1.0 / 10.0       # recovery rate
rho    = 0.9              # effectiveness of contact reduction: beta_eff = beta * (1 - rho*u2)

# ===== initial state (population normalised to 1) =====
I0 = 0.005
E0 = 0.010
R0 = 0.0
S0 = 1.0 - (E0 + I0 + R0)
x0 = np.array([S0, E0, I0, R0], dtype=float)   # [S, E, I, R]
nx, nu = 4, 2

# ===== dynamics =====
# u = [u1, u2]: u1 = vaccination rate (S -> R), u2 = contact-reduction intensity
def dyn(x, u, theta=None):
    S, E, I, R = x
    u1, u2 = u
    beta_eff = beta * (1.0 - rho * u2)

    dS = - beta_eff * S * I - u1 * S
    dE =   beta_eff * S * I - sigma * E
    dI =   sigma * E - gamma * I
    dR =   gamma * I + u1 * S
    return np.array([dS, dE, dI, dR], dtype=object)

# ===== objective: int (wI*I + wE*E + a1*u1^2 + a2*u2^2) dt =====
wI, wE = 50.0, 5.0
a1, a2 = 1.0, 10.0
def L(t, x, u, theta):
    S, E, I, R = x
    u1, u2 = u
    return (wI*I) + (wE*E) + (a1*(u1*u1)) + (a2*(u2*u2))

Phi = None  # no terminal cost

objective_builder = make_builders(
    dyn=dyn, L=L, Phi=Phi,
    quad='rk4',    # 4th-order quadrature, matching the state (see QUAD_SCHEMES)
)

# ===== control box bounds =====
u1_max = 0.20     # maximum daily vaccinated fraction
u2_max = 0.80     # maximum contact-reduction intensity
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((0.0, u1_max))  # u1
        b.append((0.0, u2_max))  # u2
    return b

# ===== assemble the problem (multi-substep RK4) =====
substepped_rk4 = partial(rk4_substeps, m_sub=10)
problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    u0=0.0,                      # start from zero control
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0, theta0=None, param_bounds_builder=None,
)
problem.quad_scheme = 'rk4'

# ====== register the constraints (canonical API) ======
# 1) path inequality: I(t) <= I_cap, i.e. h = I - I_cap <= 0 -> int L_eps(h) dt <= gamma
I_cap = 0.02
def h_peak_I(t, x, u, theta):
    return x[2] - cppad_py.a_double(I_cap)   # x[2] = I
problem.add_path_ineq(h_peak_I, eps=1e-2, gamma=0.25e-2)

# 2) integral inequality: cumulative vaccine doses <= budget.
#    The meaningful dose is int (u1*S) dt -- vaccinations delivered to S per unit
#    time -- rather than int u1 dt.
V_budget = 0.60
def q_vax(t, x, u, theta):
    S, E, I, R = x
    u1 = u[0]
    return u1 * S
problem.add_integral_ineq(qfun=q_vax, bound=V_budget, sense="<=")

__all__ = ["problem", "N", "dt", "T", "u1_max", "u2_max", "I_cap", "V_budget"]
