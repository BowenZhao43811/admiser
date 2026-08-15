# my_policy_param_problem.py
# Policy-parameter optimal control: nu = 0, the control is produced by a feedback
# policy whose coefficients are the system parameters being optimised.
import numpy as np, cppad_py
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # builds the objective_builder only

# ---------------- basic setup ----------------
T  = 15.0
N  = 100
dt = T / N

x0 = np.array([-5.0, -5.0], dtype=float)
nx, nu = 2, 0     # key point: no explicit control decisions, the policy supplies u

# Light scaling, used only to build the policy features.
SX1, SX2 = 5.0, 5.0
def scale_x(x):
    # x may hold a_double (inside the AD tape) or float (numeric rollout);
    # dtype=object keeps it compatible with a_double.
    return np.array([x[0]/SX1, x[1]/SX2], dtype=object)

# Feedback policy u = z1*x1s + z2*x2s + z3*x1s^2 + z4*x2s^2
def policy_u(x, atheta):
    x1s, x2s = scale_x(x)
    z1, z2, z3, z4 = atheta[0], atheta[1], atheta[2], atheta[3]
    return z1*x1s + z2*x2s + z3*(x1s*x1s) + z4*(x2s*x2s)

# Dynamics: the control comes from the policy, so the u passed in is ignored.
def dyn_with_policy(x, _u_ignored, atheta):
    x1, x2 = x[0], x[1]
    u = policy_u(x, atheta)
    dx1 = x2
    dx2 = -x1 + (1.4 - 0.14*(x2*x2))*x2 + u
    return np.array([dx1, dx2], dtype=object)

# Integrand L(t, x, u, theta); again u comes from the policy, not from the caller.
def L(t, x, _u_unused, atheta):
    u = policy_u(x, atheta)
    # 0.5 * (x1^2 + 0.1*x2^2 + 0.2*u^2)
    return 0.5 * ( (x[0]*x[0]) + 0.1*(x[1]*x[1]) + 0.2*(u*u) )

# no terminal cost
Phi = None

objective_builder = make_builders(
    dyn=dyn_with_policy,
    L=L,
    Phi=Phi,
    quad='rk4',         # 4th-order quadrature, matching the state (see QUAD_SCHEMES)
)

# Box bounds and initial guess for the policy coefficients theta = [z1, z2, z3, z4]
ntheta, theta0 = 4, np.zeros(4, dtype=float)
def param_bounds_builder(_):
    return [(-10.0, 10.0)] * 4

# Assemble the problem with a multi-substep RK4 integrator.
substepped_rk4 = partial(rk4_substeps, m_sub=50)

problem = OCPProblem(
    N=N, dt=dt, x0=x0,
    dyn=dyn_with_policy,           # note: the policy-driven dynamics
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,  # objective only; no constraints in this example
    control_bounds_builder=None,          # no explicit control => no box bounds
    ntheta=ntheta, theta0=theta0,
    param_bounds_builder=param_bounds_builder,
)

problem.quad_scheme = 'rk4'

__all__ = ["problem", "N", "dt", "T", "policy_u"]
