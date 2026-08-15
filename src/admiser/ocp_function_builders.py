# ocp_function_builders.py
r"""
make_builders 仅负责构建 objective_builder（\int L + Φ）。
所有约束（终端/积分/路径，等式与不等式）请使用 OCPProblem.add_* API 注册，
由 tapes.build_ad_tapes 在一次 rollout 中统一构建 eq_ad / ineq_ad。

返回值是**单个** objective_builder 函数（早期版本曾返回二元组，现已取消）。
"""

import numpy as np
import cppad_py
import inspect

from .ocp_problem_builders import DEFAULT_QUAD_SCHEME


def _bind_dyn_with_theta_ad(dyn, atheta):
    """把 dyn 统一成 f(x,u) = dyn(x,u,atheta?)（AD 版闭包）"""
    n_params = len(inspect.signature(dyn).parameters)
    if n_params == 2:
        return lambda x, u: dyn(x, u)
    return lambda x, u: dyn(x, u, atheta)

def make_builders(
    *,
    dyn,
    L=None,               # L(t, x, u, theta)；可为 None
    Phi=None,             # Phi(xT, theta)；可为 None
    # 以下两个参数保留以兼容旧调用，但不会再用于构造约束：
    terminal_eq=None,
    integral_eqs=None,
    quad: str = DEFAULT_QUAD_SCHEME,
):
    """
    返回：
      objective_builder(au, atheta, problem) -> np.array([J], dtype=object)

    quad 为本目标函数默认的子步求积方案（可选值与各自的阶数见
    ocp_integrators_utils.QUAD_SCHEMES）；若问题对象显式设置了
    problem.quad_scheme，则以后者为准——目标与所有约束必须共用同一方案，
    否则 SLSQP 看到的目标与约束建立在不同离散化上，KKT 点没有意义。
    """

    def objective_builder(au, atheta, problem):
        N, nu = problem.N, problem.nu

        # 初值（a_double）
        if hasattr(problem, "ad_initial_state") and callable(problem.ad_initial_state):
            x = problem.ad_initial_state(atheta)
        else:
            x = np.array([cppad_py.a_double(v) for v in problem.x0], dtype=object)

        t = cppad_py.a_double(0.0)
        J = cppad_py.a_double(0.0)

        f = _bind_dyn_with_theta_ad(dyn, atheta)

        def acc_cb(t_sub, x_sub, u_sub, w_sub):
            nonlocal J  # <<< 关键：必须放在函数体前部，且在首次使用 J 之前
            if L is None:
                return
            # w_sub 是求积权重，可能是 float 或 a_double；统一为 a_double
            w_ad = w_sub if isinstance(w_sub, cppad_py.a_double) else cppad_py.a_double(float(w_sub))
            _t   = t_sub if (t_sub is not None) else t
            Lval = L(_t, x_sub, u_sub, atheta)
            J    = J + w_ad * Lval

        scheme = problem.resolve_quad_scheme()

        # rollout with accumulation
        for k in range(N):
            if nu == 1:
                uk = np.array([au[k]], dtype=object)
            else:
                uk = np.asarray(au[nu*k : nu*(k+1)], dtype=object)
            dt_k = problem.dt_of_segment(k)
            x = problem.integrator(
                x, uk, dt_k, f,
                accumulate_cb=acc_cb, t0=t,
                quad=scheme,
            )
            t = t + dt_k

        # 终端代价（允许 Phi 是一个"按情况返回 None"的函数，模板即如此写法）
        if Phi is not None:
            Phi_val = Phi(x, atheta)
            if Phi_val is not None:
                J = J + Phi_val

        return np.array([J], dtype=object)

    # 把 quad 挂在函数上，供 OCPProblem.resolve_quad_scheme 读取，使目标与约束
    # tape 共用同一方案（早期版本里这个参数被 quad_scheme 无条件覆盖而失效）。
    objective_builder.quad = quad
    return objective_builder
