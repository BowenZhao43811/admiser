# my_policy_param_problem.py
# 用“新接口”改写的策略参数最优控制问题（nu=0，控制由策略产生）
import numpy as np, cppad_py
from functools import partial

from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders  # 仅构建 objective_builder

# ---------------- 基本设置 ----------------
T  = 15.0
N  = 100
dt = T / N

x0 = np.array([-5.0, -5.0], dtype=float)
nx, nu = 2, 0                     # 关键：无显式控制决策，nu=0（控制由策略产生）

# 轻量尺度化（仅用于策略的特征构造）
SX1, SX2 = 5.0, 5.0
def scale_x(x):
    # x 可能是 a_double 数组（AD tape 内）或 float 数组（数值 rollout）
    # 用 dtype=object 保持与 a_double 相容
    return np.array([x[0]/SX1, x[1]/SX2], dtype=object)

# 反馈策略 u = z1*x1s + z2*x2s + z3*x1s^2 + z4*x2s^2
def policy_u(x, atheta):
    x1s, x2s = scale_x(x)
    z1, z2, z3, z4 = atheta[0], atheta[1], atheta[2], atheta[3]
    return z1*x1s + z2*x2s + z3*(x1s*x1s) + z4*(x2s*x2s)

# 动力学（控制由策略内部给出；忽略传入的 u）
def dyn_with_policy(x, _u_ignored, atheta):
    x1, x2 = x[0], x[1]
    u = policy_u(x, atheta)
    dx1 = x2
    dx2 = -x1 + (1.4 - 0.14*(x2*x2))*x2 + u
    return np.array([dx1, dx2], dtype=object)

# 被积函数 L(t,x,u,theta) —— 这里 u 由策略给出，忽略模板传入的 u
def L(t, x, _u_unused, atheta):
    u = policy_u(x, atheta)
    # 0.5 * (x1^2 + 0.1 x2^2 + 0.2 u^2)
    return 0.5 * ( (x[0]*x[0]) + 0.1*(x[1]*x[1]) + 0.2*(u*u) )

# 无终端代价
Phi = None

# 用模板生成 objective_builder（内部会在 RK4 子步中回调累积 ∫L dt）
_ret = make_builders(
    dyn=dyn_with_policy,
    L=L,
    Phi=Phi,
    quad='rk4-mid',         # 子步采样点（mid 更稳）
)
objective_builder = _ret[0] if isinstance(_ret, tuple) else _ret

# 参数盒约束与初猜（优化策略系数 θ = [z1,z2,z3,z4]）
ntheta, theta0 = 4, np.zeros(4, dtype=float)
def param_bounds_builder(_):
    return [(-10.0, 10.0)] * 4

# 组装问题（将积分器固定为多子步 RK4）
substepped_rk4 = partial(rk4_substeps, m_sub=50)

problem = OCPProblem(
    N=N, dt=dt, x0=x0,
    dyn=dyn_with_policy,           # 注意：dyn 使用的是策略版
    integrator=substepped_rk4,     # 多子步
    nu=nu, nx=nx,
    objective_builder=objective_builder,  # 仅目标；约束用 problem.add_* 注册（本例无）
    control_bounds_builder=None,         # 无显式控制 => 无盒约束
    ntheta=ntheta, theta0=theta0,
    param_bounds_builder=param_bounds_builder,
)

# 统一：子步积分采样模式
problem.path_quad_mode = 'rk4-mid'

__all__ = ["problem", "N", "dt", "T"]
