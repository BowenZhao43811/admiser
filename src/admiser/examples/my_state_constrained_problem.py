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

# 路径不等式：h(t) ≤ 0
def hfun(t, x, u, th):
    return -8.0 * (t - 0.5)**2 + 0.5 + x[1]

# ===== 两种求解模式各自的转译参数 =====
# 单次求解（原版）：ε 与 γ 都固定，直接冷启动解这一个 NLP
EPS_SINGLE, GAMMA_SINGLE = 1e-4, 0.25e-5
# 续贯求解：ε 从 EPS_START 逐轮缩到 EPS_FINAL（= 原版的 ε，便于两者对比），
# γ 不指定 -> 自动取 T·ε/4 并随 ε 同步收缩
EPS_START, EPS_FINAL = 1e-1, 1e-4


def build_problem(continuation: bool = False) -> OCPProblem:
    """
    continuation=False：原版，γ 固定，配 OCPSolver.solve()
    continuation=True ：续贯版，省略 γ 使其自动跟随 ε，配 OCPSolver.solve_transcription()

    每次调用都返回一个全新的 problem。续贯求解会就地修改 ε/γ，所以想重跑时
    请重新 build_problem()，不要复用已经跑过的对象。
    """
    p = OCPProblem(
        N=N, dt=dt,
        x0=x0,
        u0 = u0,
        dyn=dyn, integrator=substep_rk4,
        nu=nu, nx=nx,
        objective_builder=objective_builder,
        control_bounds_builder=control_bounds_builder,
        ntheta=0,
    )
    p.quad_scheme = 'rk4'
    if continuation:
        p.add_path_ineq(hfun=hfun, eps=EPS_START)
    else:
        p.add_path_ineq(hfun=hfun, eps=EPS_SINGLE, gamma=GAMMA_SINGLE)
    return p


problem              = build_problem()                   # 原版
problem_continuation = build_problem(continuation=True)   # 续贯版

__all__ = ["problem", "problem_continuation", "build_problem",
           "N", "dt", "T", "hfun", "EPS_START", "EPS_FINAL"]
