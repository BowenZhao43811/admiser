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
    通用 OCP 求解器。

    ------------- 对外只有两个方法 -------------
    solve(...)   求解。单次还是 ε→0 续贯，由问题自己决定——在问题定义里用
                 OCPProblem.set_transcription() 声明，求解端不必区分。
    to_nlp(...)  只把 OCP 转录成 NLP（录 AD 带）并返回可求值/求梯度的对象，
                 **不做优化**。用于检查梯度、与有限差分对比等。

    ------------- 结果字典 -------------
    scipy_result : 最后一轮的 SciPy 优化结果
    U_opt        : ndarray (N*nu,)，最优控制
    theta_opt    : ndarray (ntheta,)，最优系统参数（无则 None）
    J_opt        : float，最优代价
    t_opt        : ndarray (N+1,)，时间网格
    X_opt        : ndarray (N+1, nx)，最优状态轨迹
    eq_resid     : ndarray，等式约束残差 G(z)，应 ≈ 0（无则 None）
    ineq_resid   : ndarray，不等式约束残差 C(z)，应 ≥ 0（无则 None）
    path_viol    : ndarray，各路径不等式在网格上的 max h(t)，应 ≤ 0（无则 None）
    history_cost : ndarray，最后一轮迭代过程中的代价变化
    rounds       : list[dict]，每一轮的 (eps, gamma, J_opt, max_path_viol, status,
                   nit)。single 模式下长度为 1，因此调用方不需要为两种模式写分支。

    ------------- 备注 -------------
    - solve() 不会永久改动 problem：续贯过程中 ε 只是被临时缩放，结束后恢复，
      因此同一个 problem 可以反复求解，结果一致。
    """

    def __init__(self, problem: object):
        self.problem = problem
        self.objective_ad = None
        self.eq_ad = None
        self.ineq_ad = None
        self.opt_fun = None
        self.history_cost = []

    # ================= 对外接口 =================

    def solve(self, maxiter=5000, ftol=1e-8, disp=False, verbose=None, z0=None):
        """
        求解。求解模式来自 problem.set_transcription()：

          mode="single"       ：用登记的 ε/γ 解一次
          mode="continuation" ：ε 按几何序列从大到小逐轮求解，每轮热启动

        参数
        ----
        maxiter, ftol, disp : 透传给每一轮的 SciPy SLSQP（disp 打印 SLSQP 自己的收敛信息）
        verbose : 是否逐轮打印 ε/γ/J/违反量。None 表示"续贯模式下打印，单次模式下不打印"
        z0      : 第一轮的初值，默认取 problem.initial_guess()
        """
        p = self.problem
        factors = p.eps_factors()
        if verbose is None:
            verbose = len(factors) > 1

        z = p.initial_guess() if z0 is None else np.asarray(z0, dtype=float)
        rounds, res = [], None

        for k, factor in enumerate(factors):
            # ε 只在这个 with 块内被缩放，退出即恢复；spec["eps"] 始终是用户登记的值
            with p.scaled_eps(factor=factor):
                self._build_tapes()
                res = self._solve_once(z, maxiter=maxiter, ftol=ftol, disp=disp)
                z = res["scipy_result"].x
                eps_now = [p.effective_eps(s) for s in p.path_ineq_specs]
                gam_now = [p.effective_gamma(s) for s in p.path_ineq_specs]

            viol = res["path_viol"]
            max_viol = float(np.max(viol)) if viol is not None and viol.size else float("nan")
            rounds.append(dict(eps=eps_now, gamma=gam_now, J_opt=res["J_opt"],
                               max_path_viol=max_viol,
                               status=res["scipy_result"].status,
                               nit=res["scipy_result"].nit))
            if verbose:
                e = f"{min(eps_now):.3e}" if eps_now else "—"
                print(f"[ADMISER] 第 {k+1}/{len(factors)} 轮  eps={e}  "
                      f"J={res['J_opt']:+.8g}  max h(t)={max_viol:+.3e}  "
                      f"status={res['scipy_result'].status}  nit={res['scipy_result'].nit}")

        res["rounds"] = rounds
        return res

    def to_nlp(self, eps=None):
        """
        只把 OCP 转录成 NLP 并返回可求值/求梯度的对象，不做优化。

        暴露 objective_fun / objective_grad / eq_fun / eq_jac / ineq_fun / ineq_jac，
        典型用途是拿 AD 梯度和有限差分对一下：

            nlp = OCPSolver(problem).to_nlp()
            g_ad = nlp.objective_grad(z)

        eps : 用哪个 ε 录带。None（默认）表示用**续贯第一轮实际会用的那个**
              （single 模式下即登记值本身）；给数值则所有路径约束都用它。
        """
        p = self.problem
        factor = p.eps_factors()[0]
        with p.scaled_eps(factor=factor, override=eps):
            self._build_tapes()
        return self.opt_fun

    # ================= 内部实现 =================

    def _build_tapes(self):
        """录 AD 带。ε 已经烧进带子里，所以续贯的每一轮都必须重录。"""
        z0 = self.problem.initial_guess()
        self.objective_ad, self.eq_ad, self.ineq_ad = build_ad_tapes(z0, self.problem)
        self.opt_fun = optimize_fun_class(self.objective_ad, self.eq_ad, self.ineq_ad)
        return self.opt_fun

    def _solve_once(self, z0, maxiter, ftol, disp):
        """用当前这条带子解一次 NLP。"""
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

    def _numeric_rollout(self, U, theta):
        """用 problem 的 dyn/integrator 复算最优轨迹，供出图用。"""
        p = self.problem
        N, dt, nx, nu = p.N, p.dt, p.nx, p.nu

        # 用问题提供的钩子根据 θ 生成数值初值；若没有则回退到固定 x0
        if hasattr(p, "numeric_initial_state") and callable(p.numeric_initial_state):
            x = np.asarray(p.numeric_initial_state(theta), dtype=float)
        else:
            x = np.asarray(p.x0, dtype=float)

        X = np.zeros((N+1, nx), dtype=float)
        X[0] = x
        t = np.linspace(0.0, N*dt, N+1)

        step = p.integrator
        f_num = _bind_dyn_numeric(p.dyn, theta)

        for k in range(N):
            if nu == 1:
                uk = np.array([U[k]], dtype=float)
            else:
                uk = np.asarray(U[nu*k:nu*(k+1)], dtype=float)
            x = step(x, uk, dt, f_num)  # 多子步由 integrator 内部完成
            X[k+1] = x

        return t, X
