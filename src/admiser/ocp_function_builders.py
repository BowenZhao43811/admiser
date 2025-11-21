# ocp_function_builders.py
"""
make_builders 仅负责构建 objective_builder（∫L + Φ）。
所有约束（终端/积分/路径，等式与不等式）请使用 OCPProblem.add_* API 注册，
由 tapes.build_ad_tapes 在一次 rollout 中统一构建 eq_ad / ineq_ad。

为兼容旧代码，本函数仍返回 (objective_builder, dummy_constraint_builder)，
其中 dummy_constraint_builder 总是返回空数组。
"""

import numpy as np
import cppad_py
import inspect

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
    quad: str = 'rk4-mid',
):
    """
    返回：
      objective_builder(au, atheta, problem) -> np.array([J], dtype=object)
      dummy_constraint_builder(...) -> empty (兼容旧接口)
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

        def acc_cb(t_sub, x_sub, u_sub, h_sub):
            nonlocal J  # <<< 关键：必须放在函数体前部，且在首次使用 J 之前
            if L is None:
                return
            # h_sub 可能是 float 或 a_double；统一为 a_double
            h_ad = h_sub if isinstance(h_sub, cppad_py.a_double) else cppad_py.a_double(float(h_sub))
            _t   = t_sub if (t_sub is not None) else t
            Lval = L(_t, x_sub, u_sub, atheta)
            J    = J + h_ad * Lval

        def _dt_k(_k):
            # 预留 CPET 钩子（未实现则用常数 dt）
            if hasattr(problem, "_current_dt") and callable(problem._current_dt):
                return problem._current_dt(_k)
            return cppad_py.a_double(problem.dt)

        # rollout with accumulation
        for k in range(N):
            if nu == 1:
                uk = np.array([au[k]], dtype=object)
            else:
                uk = np.asarray(au[nu*k : nu*(k+1)], dtype=object)
            dt_k = _dt_k(k)
            x = problem.integrator(
                x, uk, dt_k, f,
                accumulate_cb=acc_cb, t0=t,
                quad=getattr(problem, "path_quad_mode", quad),
            )
            t = t + dt_k

        # 终端代价
        if Phi is not None:
            Phi_val = Phi(x, atheta)
            J = J + Phi_val

        return np.array([J], dtype=object)

    return objective_builder
