# my_port_kobe_tpl.py  —— 新版（canonical constraints）

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # 仅构建 ∫L+Φ 的 objective_builder

# ========== 网格 ==========
T  = 1.0
N  = 20                   # 可先 20；收敛后再加密：200/300/400
dt = T / N

nx, nu = 6, 2

# ========== 初末状态 ==========
x0 = np.array([0.0, 22.0, 0.0, 0.0, -1.0, 0.0], dtype=float)
xT = np.array([10.0, 14.0, 0.0, 2.5, 0.0, 0.0], dtype=float)

# ========== 动力学 ==========
def dyn(x, u, theta=None):
    """
    dx1/dt = 9*x4
    dx2/dt = 9*x5
    dx3/dt = 9*x6
    dx4/dt = 9*(u1 + 17.2656*x3)
    dx5/dt = 9*u2
    dx6/dt = -9 * x2 * (u1 + 27.0756*x3 + 2*x5*x6)   # ← 按题设：乘 x2
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

# ========== 目标： 4.5 * ∫_0^1 [(x3)^2 + (x6)^2] dt ==========
def L(t, x, u, theta):
    return 4.5 * (x[2]*x[2] + x[5]*x[5])

# 仅构建目标（Φ=None）
objective_builder= make_builders(
    dyn=dyn,
    L=L,
    Phi=None,
    quad='rk4',
)

# ========== 控制盒约束 ==========
# |u1| ≤ 2.83374； -0.80865 ≤ u2 ≤ 0.71265
def control_bounds_builder(problem: OCPProblem):
    b = []
    for _ in range(problem.N):
        b.append((-2.83374,  2.83374))   # u1
        b.append((-0.80865,  0.71265))   # u2
    return b

# ========== 构造问题 ==========
substepped_rk4 = partial(rk4_substeps, m_sub=12)  # 子步多些更稳

# ---- 终端等式：x(T) - xT = 0（用 canonical API 注册）----
def terminal_eq_psi(xT_ad, atheta):
    xT_const = np.array([v for v in xT], dtype=object)
    return xT_ad - xT_const

# ========== 路径不等式（约束转译） ==========
# h1 = x4 - 2.5 ≤ 0
# h2 = -x4 - 2.5 ≤ 0
# h3 = x5 - 1.0 ≤ 0
# h4 = -x5 - 1.0 ≤ 0
def h1(t_ad, x_ad, u_ad, atheta):  # x4 <= 2.5
    return x_ad[3] - 2.5

def h2(t_ad, x_ad, u_ad, atheta):  # x4 >= -2.5
    return -x_ad[3] - 2.5

def h3(t_ad, x_ad, u_ad, atheta):  # x5 <= 1.0
    return x_ad[4] - 1.0

def h4(t_ad, x_ad, u_ad, atheta):  # x5 >= -1.0
    return -x_ad[4] - 1.0

PATH_INEQS = (h1, h2, h3, h4)

# ===== 两种求解模式各自的转译参数 =====
# 单次求解（原版）：平滑宽度 ε 与 CQ 余量 γ 都固定
EPS_SINGLE, GAMMA_SINGLE = 1e-2, 1e-3
# 续贯求解：ε 从 EPS_START 逐轮缩到 EPS_FINAL（= 原版的 ε），γ 自动取 T·ε/4。
# 四条约束共用同一组 ε/γ，续贯时一起收缩。
EPS_START, EPS_FINAL = 1e0, 1e-2


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
        dyn=dyn, integrator=substepped_rk4,
        nu=nu, nx=nx,
        objective_builder=objective_builder,
        control_bounds_builder=control_bounds_builder,
        ntheta=0, theta0=None, param_bounds_builder=None,
    )
    p.quad_scheme = 'rk4'
    p.add_terminal_eq(terminal_eq_psi)
    for hh in PATH_INEQS:
        if continuation:
            p.add_path_ineq(hh, eps=EPS_START)
        else:
            p.add_path_ineq(hh, eps=EPS_SINGLE, gamma=GAMMA_SINGLE)
    return p


problem              = build_problem()                   # 原版
problem_continuation = build_problem(continuation=True)   # 续贯版

__all__ = ["problem", "problem_continuation", "build_problem",
           "T", "N", "dt", "PATH_INEQS", "EPS_START", "EPS_FINAL"]
