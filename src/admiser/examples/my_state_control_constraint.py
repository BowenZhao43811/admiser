# my_state_constrained_problem_tpl.py  —— 新版（canonical constraints）

import numpy as np
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # 仅构建目标 ∫Ldt

# ===== 网格 =====
T  = 4.5
N  = 20
dt = T / N

nx, nu = 2, 1

# ===== 初始状态 =====
x0 = np.array([-5.0, -5.0], dtype=float)

u0 = 1.0  # 初猜：恒定控制

# ===== 动力学（3参签名，theta 此例未用）=====
def dyn(x, u, theta):
    """
    dx1/dt = x2
    dx2/dt = -x2 + u
    """
    x1, x2 = x
    uu     = u[0]              # a_double or float
    dx1 = x2
    dx2 = -x1 + x2 * (1.4 - 0.14* x2*x2) + 4 * uu
    return np.array([dx1, dx2], dtype=object)

# ===== 目标 L: ∫ (x1^2 + x2^2 + 0.005 u^2) dt =====
def L(t, x, u, theta):
    x1, x2 = x
    uu = u[0]
    return x1*x1 + uu*uu

# 终端代价/等式、积分等式：本例均无
Phi = None

# ===== 仅生成 objective_builder（约束统一走 add_* API）=====
objective_builder= make_builders(
    dyn=dyn,
    L=L,
    Phi=Phi,
    quad='rk4',   # 4 阶求积，与状态同阶（见 QUAD_SCHEMES）
)

# ===== 控制盒约束 =====
def control_bounds_builder(problem: OCPProblem):
    # 每段一个 u_k ∈ [-20, 20]
    return [(-20.0, 20.0)] * (problem.N * problem.nu)

# ===== 组装问题（多子步 RK4 提升稳定性） =====
substepped_rk4 = partial(rk4_substeps, m_sub=20)

# ===== 混合状态-控制路径不等式：h(t,x,u) = u + x1/6 ≤ 0 =====
def h(t_ad, x_ad, u_ad, atheta):
    x1, x2 = x_ad
    uu = u_ad[0]
    return uu + 1/6 * x1                   # h<=0 为可行

# ===== 两种求解模式各自的转译参数 =====
# 单次求解（原版）：ε 与 γ 都固定
EPS_SINGLE, GAMMA_SINGLE = 1e-3, 0.79e-3
# 续贯求解：ε 从 EPS_START 逐轮缩到 EPS_FINAL（= 原版的 ε），γ 自动取 T·ε/4
EPS_START, EPS_FINAL = 1e0, 1e-3


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
        dyn=dyn,
        integrator=substepped_rk4,
        nu=nu, nx=nx,
        objective_builder=objective_builder,
        control_bounds_builder=control_bounds_builder,
        ntheta=0, theta0=None, param_bounds_builder=None,
    )
    p.quad_scheme = 'rk4'
    # 注册路径不等式（Teo 的转译：∫ L_eps(h) dt ≤ γ）
    if continuation:
        p.add_path_ineq(h, eps=EPS_START)
    else:
        p.add_path_ineq(h, eps=EPS_SINGLE, gamma=GAMMA_SINGLE)
    return p


problem              = build_problem()                   # 原版
problem_continuation = build_problem(continuation=True)   # 续贯版

__all__ = ["problem", "problem_continuation", "build_problem",
           "N", "dt", "T", "h", "EPS_START", "EPS_FINAL"]
