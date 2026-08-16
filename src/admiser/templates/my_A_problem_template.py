# my_A_problem_template.py
# General OCP problem template: the objective and the constraints are decoupled,
# and all constraints are registered in canonical form.
import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders   # builds the objective_builder only

# ============= 1) grid and dimensions =============
T  = 1.0
N  = 100
dt = T / N
nx = 2
nu = 1

# ============= 2) initial state / optional terminal target =============
x0 = np.zeros(nx, dtype=float)
u0 = None  # control initial guess; None is fine if you have no preference
# A terminal equality x(T) = xT is registered with problem.add_terminal_eq(...) below.
xT = None

# ============= 3) dynamics (must tolerate theta=None) =============
def dyn(x, u, theta=None):
    """
    Example:
      x = [x1, x2], u = [u1]
      x1' = x2
      x2' = -x1 + u1
    """
    x1, x2 = x
    u1 = u[0]
    dx1 = x2
    dx2 = -x1 + u1
    return np.array([dx1, dx2], dtype=object)

# ============= 4) objective pieces (L and Phi) =============
def L(t, x, u, theta):
    """Example: L = x1^2 + x2^2 + 1e-3 * u^2"""
    return x[0]*x[0] + x[1]*x[1] + 1e-3 * (u[0]*u[0])

def Phi(xT_ad, theta):
    """Return None when there is no terminal cost; this example has none."""
    return None

# Only the objective builder; constraints go through the add_* API.
objective_builder = make_builders(
    dyn=dyn,
    L=L,           # may be None
    Phi=Phi,       # may be None
    quad='rk4'     # substep quadrature scheme, orders 1 to 4; see admiser.QUAD_SCHEMES
)

# ============= 5) optional: theta-dependent initial state x0 = x0(theta) =============
def x0_from_theta_ad(atheta, problem):
    """Example: x0 = [theta1, 0]. Omit both hooks if you do not optimise parameters."""
    if atheta is None:
        return np.array([0.0, 0.0], dtype=object)
    return np.array([atheta[0], 0.0], dtype=object)

def x0_from_theta_numeric(theta, problem):
    if theta is None:
        return x0.copy()
    return np.array([float(theta[0]), 0.0], dtype=float)

# ============= 6) control box bounds =============
def control_bounds_builder(problem: OCPProblem):
    """Example: |u1| <= 10"""
    return [(-10.0, 10.0)] * (problem.N * problem.nu)

# ============= 7) optional system parameters theta =============
ntheta = 0
theta0 = None

def param_bounds_builder(problem: OCPProblem):
    """When ntheta > 0, return one box bound per parameter. Example: theta1 in [0, 1]."""
    return [(0.0, 1.0)]

# ============= 8) assemble the problem =============
substepped_rk4 = partial(rk4_substeps, m_sub=10)

problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,                         # overridden by the hooks below if x0 depends on theta
    u0=u0,
    dyn=dyn,
    integrator=substepped_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=ntheta,
    theta0=theta0,
    param_bounds_builder=param_bounds_builder if ntheta > 0 else None,
    x0_from_theta_ad=x0_from_theta_ad if ntheta > 0 else None,
    x0_from_theta_numeric=x0_from_theta_numeric if ntheta > 0 else None,
)
problem.quad_scheme = 'rk4'

# ============= 9) register the constraints =============
# WARNING: the six kinds below are a MENU, and they are not mutually feasible
# (9.1 demands x1(T) = 1 while 9.6 demands x1(T) <= 0.5). This template enables
# only the self-consistent set 9.1 + 9.4 + 9.5; the rest are kept as comments.
# Pick what your own problem needs -- do not switch them all on at once.

# 9.1 terminal equality: x(T) = xT
def terminal_eq_psi(xT_ad, atheta):
    xT_const = np.array([v for v in [1.0, 0.0]], dtype=object)
    return xT_ad - xT_const   # vector equality = 0
problem.add_terminal_eq(terminal_eq_psi)

# 9.2 integral equality: int q(t, x, u, theta) dt = target
# def q_int(t, x, u, th): return x[0]
# problem.add_integral_eq(qfun=q_int, target=1.0)

# 9.3 path equality: h(t, x, u, theta) = 0
# - mode 'L2' : int h^2 dt = 0 (recommended)
# - mode 'abs': int smooth_abs_eps(h) dt = 0; eps_abs shrinks along with eps
#               during a continuation solve
# def h_eq(t, x, u, th): return x[1]
# problem.add_path_eq(hfun=h_eq, mode="L2", eps_abs=1e-6)

# 9.4 integral inequality: int q dt <=/>= bound
def q_budget(t, x, u, th): return u[0]*u[0]   # example: a control-energy budget
problem.add_integral_ineq(qfun=q_budget, bound=50.0, sense="<=")

# 9.5 path inequality (constraint transcription): h(t,x,u,theta) <= 0
#     -> int L_eps(h) dt <= gamma
# Omitting gamma gives the value consistent with eps, gamma = T*eps/4
# (see OCPProblem.auto_gamma).
# Never write gamma=0.0: that forces h(t) <= -eps, a strictly interior solution
# that is both conservative and hard to converge.
# eps CARRIES UNITS -- scale it to the magnitude of your own h. When unsure, let
# a continuation solve walk it down from a large value (section 10).
def h_ineq(t, x, u, th): return x[1] - 3.0   # example: keep x2 below 3
problem.add_path_ineq(hfun=h_ineq, eps=1e-2)

# 9.6 terminal inequality: phi(xT, theta) <= 0 or >= 0
# def phi_T(xT_ad, th): return xT_ad[0] - 0.5
# problem.add_terminal_ineq(phi=phi_T, sense="<=")

# ============= 10) solve mode (declared here; the driver just calls solve()) =============
# mode="single" (the default -- omit this call entirely to get it)
#     Solve once with the eps/gamma registered by add_path_ineq above.
# mode="continuation"
#     eps -> 0 continuation: the registered eps is the FINAL value, and the rounds
#     start from eps/shrink^(n_rounds-1) and shrink each round, warm-starting from
#     the previous solution. A large eps is smooth and easy but loose; a small eps
#     approaches the true constraint but also a nondifferentiable hinge, so walking
#     it down gets both. Below, eps runs 1e1 -> 1e0 -> 1e-1 -> 1e-2.
problem.set_transcription(mode="continuation", n_rounds=4, shrink=0.1)

# ============= 11) optional: the time-scaling transform (CPET) =============
# Off by default. Enabling it makes the SEGMENT DURATIONS decision variables, so
# the optimiser chooses where the segments end, not just the control value on
# each one. The decision vector becomes  z = [U (N*nu) ; theta (ntheta) ; tau (N)].
#
# Two situations where it pays off:
#
#   Bang-bang solutions. With a fixed uniform grid the switching instant can only
#   land on a grid point, so resolving it costs a fine grid everywhere. On the
#   my_bang_bang problem, CPET with N=8 (24 variables) beats a uniform grid with
#   N=84 (168 variables).
#
#   Free terminal time. Leave total_time unset and make the objective the elapsed
#   time, L = 1.0, since int 1 dt = sum(tau). No extra state or control needed.
#
# Fixed horizon instead: pass total_time=T, which registers sum(tau) = T.
#
# problem.set_time_scaling(tau0=dt, tau_min=0.0, tau_max=None, total_time=T)

# ============= 12) optional: automatic scaling =============
# ON by default -- you do not need this line at all. Write L and your constraints
# in whatever units are natural for your problem; the solver normalises them
# internally and converts everything it reports back into your units.
#
# Why it is on: SLSQP's ftol compares the ABSOLUTE change in the objective. With
# an objective of order 1e6, ftol=1e-9 asks for 1e-15 relative accuracy, below
# machine precision, so the line search gives up after a few iterations -- at a
# point that may be infeasible, and with no error raised. Normalising turns that
# absolute test into an effectively relative one. Constraint rows are scaled by
# 1/||grad g_i|| for a different reason: to condition the QP subproblem.
#
# solve() prints one line saying what factors it used. res["scaling"] holds them.
#
# problem.set_scaling(objective="auto", constraints="auto")   # the default
# problem.set_scaling(objective="none", constraints="none")   # switch it all off
# problem.set_scaling(objective=1e-6)                         # pick the factor yourself

# ============= 13) exports =============
__all__ = ["problem", "N", "dt", "T"]
