# ocp_solver.py

import inspect

import numpy as np
from scipy.optimize import minimize

from .ocp_ADfunction_tapes import build_ad_tapes
from .ocp_ADgradient_builders import optimize_fun_class


def _bind_dyn_numeric(dyn, theta):
    """
    把 dyn 统一成 f(x,u)（数值版闭包）。必须和 AD 侧的 _bind_dyn_with_theta_ad
    用同样的规则识别 2 参数 / 3 参数签名，否则只支持 dyn(x,u) 的问题会在优化
    全部跑完、复算轨迹时才抛 TypeError。
    """
    n_params = len(inspect.signature(dyn).parameters)
    if n_params == 2:
        return lambda x, u: np.asarray(dyn(x, u), dtype=float)
    return lambda x, u: np.asarray(dyn(x, u, theta), dtype=float)


class OCPSolver:
    """
    通用 OCP 求解器：
    - 输入一个 OCPProblem
    - 自动构建 AD tape
    - 自动设置 bounds / 约束
    - 调 SciPy.SLSQP
    - 返回最优控制、轨迹等（内部数值推演，不用外部 rollout）
    ------------- 方法 -------------
    build(): 构建 AD tape 和封装器
    solve(maxiter=5000, ftol=1e-8, disp=True, z0=None): 求解问题，返回结果字典
    solve_transcription(...): 约束转译的 ε→0 续贯求解（外层热启动循环）
    ------------- 结果字典 -------------
    scipy_result: SciPy 优化结果
    U_opt: ndarray (N*nu,), 最优控制
    theta_opt: ndarray (ntheta,), 最优系统参数（如有）
    J_opt: float, 最优代价
    t_opt: ndarray (N+1,), 最优时间轨迹
    X_opt: ndarray (N+1, nx), 最优状态轨迹
    eq_resid: ndarray, 最优等式约束残差（如有）
    ineq_resid: ndarray, 最优不等式约束残差 C(z)≥0（如有）
    path_viol: ndarray, 各路径不等式在网格上的 max h(t)（如有），>0 即为实际违反
    history_cost: list of float, 迭代过程中代价变化
    ------------- 备注 -------------
    - 内部数值推演使用 problem 提供的 dyn 和 integrator
    - 支持系统参数 θ 和由 θ 推初值的钩子
    - 支持等式和不等式约束
    """

    def __init__(self, problem: object):
        self.problem = problem
        self.objective_ad = None
        self.eq_ad = None
        self.ineq_ad = None
        self.opt_fun = None
        self.history_cost = []

    def build(self):
        """
        创建 objective_ad / constraint_ad 与封装器
        """
        z0 = self.problem.initial_guess()
        obj_ad, eq_ad, ineq_ad = build_ad_tapes(z0, self.problem)
        self.objective_ad  = obj_ad
        self.eq_ad         = eq_ad
        self.ineq_ad       = ineq_ad
        self.opt_fun = optimize_fun_class(obj_ad, eq_ad, ineq_ad)
        return self

    def solve(self, maxiter=5000, ftol=1e-8, disp=True, z0=None):
        """
        求解一次 NLP。z0 用于热启动（续贯求解时传上一轮的解），默认取 problem 的初猜。
        """
        if self.opt_fun is None:
            raise RuntimeError("Call build() first.")
        z0 = self.problem.initial_guess() if z0 is None else np.asarray(z0, dtype=float)
        bounds = self.problem.make_bounds()

        constraints = []
        if self.eq_ad is not None:
            constraints.append({'type': 'eq', 'fun': self.opt_fun.eq_fun, 'jac': self.opt_fun.eq_jac})
        if self.ineq_ad is not None:
            constraints.append({'type': 'ineq', 'fun': self.opt_fun.ineq_fun, 'jac': self.opt_fun.ineq_jac})

        self.history_cost = []
        res = minimize(
            fun=self.opt_fun.objective_fun, x0=z0, method='SLSQP',
            jac=self.opt_fun.objective_grad, bounds=bounds, constraints=constraints,
            callback=lambda zk: self.history_cost.append(self.opt_fun.objective_fun(zk)),
            options=dict(maxiter=maxiter, ftol=ftol, disp=disp),
        )

        U_opt, theta_opt = self.problem.split_decision(res.x)
        J_opt = self.opt_fun.objective_fun(res.x)
        eq_resid   = self.opt_fun.eq_fun(res.x)   if self.eq_ad   is not None else None
        ineq_resid = self.opt_fun.ineq_fun(res.x) if self.ineq_ad is not None else None
        t_opt, X_opt = self._numeric_rollout(U_opt, theta_opt)

        return dict(
            scipy_result=res, U_opt=U_opt, theta_opt=theta_opt, J_opt=J_opt,
            t_opt=t_opt, X_opt=X_opt, eq_resid=eq_resid, ineq_resid=ineq_resid,
            path_viol=self._path_violation(t_opt, X_opt, U_opt, theta_opt),
            history_cost=np.array(self.history_cost),
        )

    # ---------- 约束转译的 ε→0 续贯求解 ----------
    def solve_transcription(self, eps0=None, eps_min=1e-6, shrink=0.1,
                            maxiter=5000, ftol=1e-8, disp=False, verbose=True,
                            **solve_kwargs):
        """
        Teo 约束转译的标准外层循环：从一个较大的（数值上好解的）ε 出发，解一次
        NLP，然后按 shrink 收缩 ε 并**热启动**重解，直到 ε ≤ eps_min。

        为什么需要它：ε 越小，L_eps 越接近不可微的 hinge，直接从小 ε 冷启动往往
        让 SLSQP 在约束边界上卡死；而 ε 大时问题光滑好解但约束被过度松弛。续贯
        求解兼顾两者。ε 是以常数形式录进 tape 的，因此每轮需要重新 build()。

        参数：
          eps0    : 起始 ε；None 表示沿用各路径约束当前登记的 ε
          eps_min : 终止 ε（最小的那个路径约束 ε 达到即停）
          shrink  : 每轮乘性收缩因子（0<shrink<1）
          verbose : 打印每轮的 ε/γ/J/违反量
          其余参数透传给 solve()

        gamma 登记为 auto（add_path_ineq 未显式给 gamma）的约束，每轮按
        γ = T·ε/4 同步更新；显式给定的 gamma 保持不变。
        path_eq 中 mode='abs' 的 eps_abs 也按同样节奏收缩。

        返回：最后一轮的结果字典，另含
          continuation: 每轮 (eps, gamma, J_opt, max_path_viol, status) 的列表
        """
        p = self.problem
        if not (0.0 < shrink < 1.0):
            raise ValueError(f"shrink 必须在 (0,1) 内，当前 {shrink}")
        if not p.path_ineq_specs and not any(s["mode"] == "abs" for s in p.path_eq_specs):
            raise ValueError(
                "问题里没有登记任何需要 ε 的约束（path_ineq / mode='abs' 的 path_eq），"
                "无需续贯求解，请直接用 solve()。"
            )

        if eps0 is not None:
            for spec in p.path_ineq_specs:
                spec["eps"] = float(eps0)
            for spec in p.path_eq_specs:
                if spec["mode"] == "abs":
                    spec["eps_abs"] = float(eps0)

        history, z, res = [], None, None
        while True:
            for spec in p.path_ineq_specs:
                if spec.get("gamma_auto", False):
                    spec["gamma"] = p.auto_gamma(spec["eps"])

            self.build()
            res = self.solve(maxiter=maxiter, ftol=ftol, disp=disp, z0=z)
            z = res["scipy_result"].x

            eps_now = [s["eps"] for s in p.path_ineq_specs]
            viol = res["path_viol"]
            max_viol = float(np.max(viol)) if viol is not None and viol.size else float("nan")
            history.append(dict(
                eps=list(eps_now),
                gamma=[s["gamma"] for s in p.path_ineq_specs],
                J_opt=res["J_opt"], max_path_viol=max_viol,
                status=res["scipy_result"].status, nit=res["scipy_result"].nit,
            ))
            if verbose:
                e = min(eps_now) if eps_now else float("nan")
                print(f"[ADMISER] eps={e:.3e}  J={res['J_opt']:+.8g}  "
                      f"max h(t)={max_viol:+.3e}  status={res['scipy_result'].status} "
                      f"nit={res['scipy_result'].nit}")

            # 相对容差比较：shrink 后的 1e-3*0.1 = 1.0000000000000002e-04 严格大于
            # 1e-4，用精确比较会白白多解一轮。
            if not eps_now or min(eps_now) <= eps_min * (1.0 + 1e-9):
                break
            for spec in p.path_ineq_specs:
                spec["eps"] = max(float(spec["eps"]) * shrink, float(eps_min))
            for spec in p.path_eq_specs:
                if spec["mode"] == "abs":
                    spec["eps_abs"] = max(float(spec["eps_abs"]) * shrink, float(eps_min))

        res["continuation"] = history
        return res

    # ---------- 内部：路径约束的真实违反量 ----------
    def _path_violation(self, t, X, U, theta):
        """
        在网格点上求各路径不等式的 max h(t)。这是转译效果的直接诊断量：
        ineq_resid 只告诉你 γ - ∫L_eps 是否 ≥0，而这里给出原始约束 h(t) ≤ 0
        实际被违反了多少。
        """
        p = self.problem
        if not p.path_ineq_specs:
            return None
        nu = p.nu
        out = []
        for spec in p.path_ineq_specs:
            worst = -np.inf
            for k in range(len(t)):
                kk = min(k, p.N - 1)
                uk = (np.asarray(U[nu*kk: nu*(kk+1)], dtype=float) if nu > 0
                      else np.empty(0, dtype=float))
                try:
                    hv = np.atleast_1d(np.asarray(spec["hfun"](t[k], X[k], uk, theta), dtype=float))
                except Exception:
                    return None          # 用户的 hfun 只支持 AD 类型时静默跳过诊断
                worst = max(worst, float(np.max(hv)))
            out.append(worst)
        return np.asarray(out, dtype=float)

    # ---------- 内部：数值推演，使用 θ → x0(θ) 的钩子 ----------
    def _numeric_rollout(self, U, theta):
        p = self.problem
        N, dt, nx, nu = p.N, p.dt, p.nx, p.nu

        # 1) 用问题提供的钩子根据 θ 生成"数值初值"；若没有则回退到固定 x0
        if hasattr(p, "numeric_initial_state") and callable(p.numeric_initial_state):
            x = np.asarray(p.numeric_initial_state(theta), dtype=float)
        else:
            x = np.asarray(p.x0, dtype=float)

        X = np.zeros((N+1, nx), dtype=float)
        X[0] = x
        t = np.linspace(0.0, N*dt, N+1)

        step = p.integrator
        f_num = _bind_dyn_numeric(p.dyn, theta)

        # 2) 按段推进
        for k in range(N):
            if nu == 1:
                uk = np.array([U[k]], dtype=float)
            else:
                uk = np.asarray(U[nu*k:nu*(k+1)], dtype=float)
            x = step(x, uk, dt, f_num)  # 多子步由 integrator 内部完成
            X[k+1] = x

        return t, X
