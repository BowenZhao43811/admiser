# my_free_terminal_time.py -- minimum-time problem solved with the time-scaling transform
#
# Minimise the time needed to steer the system to a terminal heading condition.
#
# Before the package supported CPET this had to be rewritten by hand: an extra
# state x4 and an extra control u2 were introduced so that every dynamics equation
# read dx/dt = u2 * f(x, u), x4 accumulated u2, and the objective became x4(T).
# That is precisely the time-scaling transform, done manually. Now the transform is
# built in: state and control keep their natural meaning, the durations tau_k are
# the decision variables, and the objective is simply the elapsed time.

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders

# ===== grid =====
# With the transform, N is the number of segments and dt is only the initial guess
# for each duration; the horizon itself is optimised, not fixed.
N  = 10
dt = 0.1

nx, nu = 3, 1

# ===== initial state =====
x0 = np.array([np.pi/2, 4.0, 0.0], dtype=float)   # [heading, x, y]

u0 = 1.0   # initial guess for the turn rate

# ===== dynamics, in real time =====
# dx1/dt = u1          (heading rate)
# dx2/dt = cos(x1)     (unit-speed motion)
# dx3/dt = sin(x1)
def dyn(x, u, theta=None):
    x1, x2, x3 = x
    u1 = u[0]
    return np.array([u1, np.cos(x1), np.sin(x1)], dtype=object)

# ===== objective: minimise the elapsed time =====
# int 1 dt == sum(tau), so a constant integrand IS the horizon. No extra state,
# no extra control.
def L(t, x, u, theta):
    return 1.0

objective_builder = make_builders(dyn=dyn, L=L, Phi=None, quad='rk4')

# ===== control box bounds =====
def control_bounds_builder(problem: OCPProblem):
    return [(-2.0, 2.0)] * (problem.N * problem.nu)   # |turn rate| <= 2

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
# tau_min is kept slightly positive here: the dynamics are fine at tau = 0, but
# giving every segment a little room keeps the first iterations better behaved.
# total_time is left unset, which is what makes the horizon free.
problem.set_time_scaling(tau0=dt, tau_min=1e-4, tau_max=5.0)

# ===== terminal equalities: x2(T) = 0, x3(T) = 0 =====
def terminal_eq_psi(xT_ad, theta):
    return np.array([
        xT_ad[1] - 0.0,   # x2(T) = 0
        xT_ad[2] - 0.0,   # x3(T) = 0
    ], dtype=object)

problem.add_terminal_eq(terminal_eq_psi)

__all__ = ["problem", "N", "dt"]
