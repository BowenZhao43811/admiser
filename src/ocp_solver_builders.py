# ocp_solver.py

import numpy as np
from scipy.optimize import minimize

from admiser import build_ad_tapes
from admiser import optimize_fun_class

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
    solve(maxiter=5000, ftol=1e-8, disp=True): 求解问题，返回结果字典
    ------------- 结果字典 -------------
    scipy_result: SciPy 优化结果
    U_opt: ndarray (N*nu,), 最优控制
    theta_opt: ndarray (ntheta,), 最优系统参数（如有）
    J_opt: float, 最优代价
    t_opt: ndarray (N+1,), 最优时间轨迹
    X_opt: ndarray (N+1, nx), 最优状态轨迹
    eq_resid: ndarray, 最优等式约束残差（如有）
    ineq_resid: ndarray, 最优不等式约束残差（如有）
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
        # self.opt_fun = optimize_fun_class(obj_ad, eq_ad, ineq_ad)
        self.opt_fun = optimize_fun_class(obj_ad, eq_ad, ineq_ad)
        return self

    def solve(self, maxiter=5000, ftol=1e-8, disp=True):
        if self.opt_fun is None:
            raise RuntimeError("Call build() first.")
        z0 = self.problem.initial_guess()
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
            history_cost=np.array(self.history_cost),
        )

    # ---------- 内部：数值推演，使用 θ → x0(θ) 的钩子 ----------
    def _numeric_rollout(self, U, theta):
        p = self.problem
        N, dt, nx, nu = p.N, p.dt, p.nx, p.nu

        # 1) 用问题提供的钩子根据 θ 生成“数值初值”；若没有则回退到固定 x0
        if hasattr(p, "numeric_initial_state") and callable(p.numeric_initial_state):
            x = np.asarray(p.numeric_initial_state(theta), dtype=float)
        else:
            # 回退：固定初值（注意：若你的问题是“初值依赖 θ”，请实现 numeric_initial_state）
            x = np.asarray(p.x0, dtype=float)

        X = np.zeros((N+1, nx), dtype=float)
        X[0] = x
        t = np.linspace(0.0, N*dt, N+1)

        step = p.integrator

        # dyn 的数值版本（把返回值转为 float，integrator 可处理 float）
        def f_num(x_, u_):
            return np.asarray(p.dyn(x_, u_, theta), dtype=float)

        # 2) 按段推进
        for k in range(N):
            if nu == 1:
                uk = np.array([U[k]], dtype=float)
            else:
                uk = np.asarray(U[nu*k:nu*(k+1)], dtype=float)
            x = step(x, uk, dt, f_num)  # 多子步由 integrator 内部完成
            X[k+1] = x

        return t, X
