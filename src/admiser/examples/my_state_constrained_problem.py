import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders

T = 1.0
N = 40
dt = T / N
nu, nx = 1, 2

x0 = np.array([0.0, -1.0], dtype=float)

u0 = 1.0  # 初猜：恒定控制

def dyn(x, u, theta=None):
    x1, x2 = x
    uu = u[0]
    dx1 = x2
    dx2 = -x2 + uu
    return np.array([dx1, dx2], dtype=object)

def L(t, x, u, theta):
    return (x[0]*x[0]) + (x[1]*x[1]) + 0.005*(u[0]*u[0])

objective_builder= make_builders(dyn=dyn, L=L, Phi=None, quad='rk4')

def control_bounds_builder(problem: OCPProblem):
    return [(-20.0, 20.0)] * (problem.N * problem.nu)

substep_rk4 = partial(rk4_substeps, m_sub=10)
problem = OCPProblem(
    N=N, dt=dt,
    x0=x0,
    u0 = u0,
    dyn=dyn, integrator=substep_rk4,
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0,
)
problem.quad_scheme = 'rk4'

# 路径不等式：h(t) ≤ 0
def hfun(t, x, u, th):
    return -8.0 * (t - 0.5)**2 + 0.5 + x[1]

# 省略 gamma 即按 γ = T·ε/4 自动取值，并在续贯模式下随 ε 同步收缩
problem.add_path_ineq(hfun=hfun, eps=1e-4)

# ===== 求解模式：写在这里，求解端只管调 OCPSolver(problem).solve() =====
# mode="single"       ：用上面登记的 ε=1e-4 直接解一次
# mode="continuation" ：ε 视为终点，按 1e-1 → 1e-2 → 1e-3 → 1e-4 逐轮热启动求解
problem.set_transcription(mode="continuation", n_rounds=4, shrink=0.1)

__all__ = ["problem", "N", "dt", "T", "hfun"]
