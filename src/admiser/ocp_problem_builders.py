# ocp_problem.py
import numpy as np
import cppad_py

from .ocp_integrators_utils import resolve_quad_name

#: 包级默认求积方案：与状态同为 4 阶，且不增加动力学求值次数
DEFAULT_QUAD = 'rk4'


class OCPProblem:
    """
    统一的 OCP 问题容器。
    - 目标函数由 objective_builder(U[a_double], theta[a_double], problem) 构建
    - 约束通过 add_* 接口注册（canonical 形式）并由 tapes 统一构建：
        等式:  G(z) = 0
        不等式: C(z) >= 0
    - 动力学与积分由 integrator 处理（需支持 accumulate_cb, t0, quad）
    - 控制初猜 u0：支持多种形状并自动展开为长度 N*nu 的向量
        * 标量：同一值用于所有步与所有控制分量
        * (nu,)：每步相同的控制向量
        * (N,)：仅当 nu==1 时有效（每步一个标量控制）
        * (N, nu)：逐步控制矩阵
        * (N*nu,)：已扁平向量
      若未提供 u0，则优先用控制盒约束的“中点”作为初猜；无界处取 0。
    """

    def __init__(
        self,
        *,
        N: int,
        dt: float,
        x0: np.ndarray,
        u0=None,                           # 可选：控制初猜（见上面支持的形状）
        dyn=None,                          # dyn(x, u, theta=None) -> np.array(dtype=object)
        integrator=None,                   # integrator(x, u, dt_k, f, accumulate_cb=None, t0=None, quad='rk4-mid')
        nu: int,
        nx: int,
        objective_builder=None,            # 必填
        constraint_builder=None,           # 兼容旧接口：可为 None（新架构建议不用）
        control_bounds_builder=None,       # -> list[(low, high)] of length N*nu
        ntheta: int = 0,
        theta0: np.ndarray | None = None,
        param_bounds_builder=None,         # -> list[(low, high)] length ntheta
        x0_from_theta_ad=None,             # 可选：初值依赖 theta 的 a_double 版本
        x0_from_theta_numeric=None,        # 可选：数值版
    ):
        self.N  = int(N)
        self.dt = float(dt)
        self.x0 = np.asarray(x0, dtype=float)
        self.u0 = None if u0 is None else np.asarray(u0, dtype=float)

        self.nu = int(nu)
        self.nx = int(nx)

        self.dyn = dyn
        self.integrator = integrator

        self.objective_builder  = objective_builder
        # 旧接口保留以兼容历史代码；建议使用 add_* 系列注册约束
        self.constraint_builder = constraint_builder

        self.control_bounds_builder = control_bounds_builder

        self.ntheta = int(ntheta) if ntheta is not None else 0
        self.theta0 = None if theta0 is None else np.asarray(theta0, dtype=float)
        self.param_bounds_builder = param_bounds_builder

        # 初值依赖 theta（可选）
        self._x0_from_theta_ad      = x0_from_theta_ad
        self._x0_from_theta_numeric = x0_from_theta_numeric

        # 统一的规范化约束注册表
        self.term_eq_specs   = []  # ψ_i(xT,θ) = 0
        self.term_ineq_specs = []  # φ_i(xT,θ) ≤ 0 / ≥ 0
        self.int_eq_specs    = []  # ∫ q_i dt  = b_i
        self.int_ineq_specs  = []  # ∫ q_i dt ≤/≥ b_i
        self.path_eq_specs   = []  # 路径等式 h(t)=0 → ∫ h^2 dt = 0 或 ∫ smooth|h| dt = 0
        self.path_ineq_specs = []  # 路径不等式 h≤0 → ∫ L_eps(h) dt ≤ γ

        # 子步积分采样模式。None = 未指定，由 objective_builder 自带的 quad 决定
        # （见 resolve_quad_mode）；显式赋值则覆盖之，对目标与全部约束统一生效。
        self.path_quad_mode = None

    # ---------- 求积方案解析 ----------
    def resolve_quad_mode(self) -> str:
        """
        全局唯一的求积方案解析，目标与所有约束都必须经由此处取值，优先级：
          1. problem.path_quad_mode（显式赋值）
          2. make_builders(quad=...) 记在 objective_builder.quad 上的值
          3. 包级默认 DEFAULT_QUAD

        返回的是**规范名**（别名如 'rk4-mid' 会被归一到 'midpoint'），因此
        下游可以直接按规范名比较。不认识的名字在此报错，而不是拖到积分器里。
        """
        mode = getattr(self, "path_quad_mode", None)
        if mode is None:
            mode = getattr(self.objective_builder, "quad", None)
        if mode is None:
            mode = DEFAULT_QUAD
        return resolve_quad_name(mode)

    # ---------- 规范化约束注册 API ----------
    def add_terminal_eq(self, psi):
        """psi(xT, theta) -> a_double 或 a_double 向量"""
        self.term_eq_specs.append(dict(psi=psi))

    def add_terminal_ineq(self, phi, sense: str = "<="):
        """phi(xT, theta) <= 0 或 >= 0，sense in {'<=','>='}"""
        assert sense in ("<=", ">=")
        self.term_ineq_specs.append(dict(phi=phi, sense=sense))

    def add_integral_eq(self, qfun, target: float):
        """∫ q dt = target；qfun(t,x,u,theta)->a_double"""
        self.int_eq_specs.append(dict(qfun=qfun, target=float(target)))

    def add_integral_ineq(self, qfun, bound: float, sense: str = "<="):
        """∫ q dt ≤/≥ bound；sense in {'<=','>='}"""
        assert sense in ("<=", ">=")
        self.int_ineq_specs.append(dict(qfun=qfun, bound=float(bound), sense=sense))

    def auto_gamma(self, eps: float) -> float:
        """
        约束转译中与 eps 相容的 γ 取法：γ = T·ε/4，T = N·dt。

        依据：L_eps(h) - max(h,0) ∈ [0, ε/4]，在 h ≤ 0 上 L_eps(h) 的最大值恰为
        L_eps(0) = ε/4。因此
          * 任何真正满足 h(t) ≤ 0 的轨迹都满足 ∫L_eps dt ≤ T·ε/4 —— 转译后的问题
            不会把原问题的可行解（包括最优解）排除在外；
          * 反过来 ∫max(h,0) dt ≤ ∫L_eps dt ≤ γ = T·ε/4，即违反量的 L¹ 范数被
            T·ε/4 控制，ε→0 时收敛到原约束。
        取 γ=0 则强制 L_eps ≡ 0，等价于 h(t) ≤ -ε（严格内点），既过于保守又常常
        让 SLSQP 在边界上失去下降方向。
        """
        return float(self.N * self.dt * float(eps) / 4.0)

    def add_path_ineq(self, hfun, eps: float, gamma: float | None = None):
        """
        h(x,u,θ) ≤ 0  —>  ∫ L_eps(h) dt ≤ γ

        gamma=None（默认）表示按 auto_gamma(eps) = T·ε/4 自动取值，并在续贯
        求解（OCPSolver.solve_transcription）收缩 ε 时随之更新；显式传入数值则
        固定不变。
        """
        auto = gamma is None
        self.path_ineq_specs.append(dict(
            hfun=hfun, eps=float(eps),
            gamma=self.auto_gamma(eps) if auto else float(gamma),
            gamma_auto=auto,
        ))

    def add_path_eq(self, hfun, mode: str = "L2", eps_abs: float = 1e-6):
        """
        路径等式 h(t)=0：
        - mode='L2' : ∫ h^2 dt = 0（推荐）
        - mode='abs': ∫ smooth_abs_eps(h) dt = 0
        """
        assert mode in ("L2", "abs")
        self.path_eq_specs.append(dict(hfun=hfun, mode=mode, eps_abs=float(eps_abs)))

    # ---------- 基础工具 ----------
    def has_params(self) -> bool:
        return self.ntheta > 0

    def dt_of_segment(self, k: int):
        """
        第 k 段的时长（a_double）。目标 tape 与约束 tape 必须都走这一个入口，
        否则将来接入时间尺度变换（CPET）时两者会静默用上不同的时间网格。
        子类/用户可实现 _current_dt(k) 返回 a_double 以支持变步长。
        """
        if callable(getattr(self, "_current_dt", None)):
            return self._current_dt(k)
        return cppad_py.a_double(self.dt)

    def validate(self) -> None:
        """在录带前做一次结构检查，把 CppAD/SciPy 抛出的天书换成可读的报错。"""
        if not callable(self.objective_builder):
            raise ValueError(
                "OCPProblem.objective_builder 未设置。请用 make_builders(dyn=..., L=..., Phi=...) "
                "生成后传给 OCPProblem(objective_builder=...)。"
            )
        if not callable(self.integrator):
            raise ValueError("OCPProblem.integrator 未设置，例如 partial(rk4_substeps, m_sub=10)。")
        if not callable(self.dyn):
            raise ValueError("OCPProblem.dyn 未设置。")
        if self.N <= 0 or self.dt <= 0:
            raise ValueError(f"需要 N>0 且 dt>0，当前 N={self.N}, dt={self.dt}.")
        if self.x0.shape != (self.nx,):
            raise ValueError(f"x0 形状 {self.x0.shape} 与 nx={self.nx} 不符。")

        n_z = self.N * self.nu + (self.ntheta if self.has_params() else 0)
        bnds = self.make_bounds()
        if len(bnds) != n_z:
            raise ValueError(
                f"盒约束个数 {len(bnds)} 与决策变量维数 {n_z} (=N*nu+ntheta) 不符；"
                "请检查 control_bounds_builder / param_bounds_builder 的返回长度。"
            )
        if self.initial_guess().size != n_z:
            raise ValueError(
                f"初猜维数 {self.initial_guess().size} 与决策变量维数 {n_z} 不符（检查 u0 / theta0）。"
            )
        if self.has_params() and self.theta0 is None:
            print("[ADMISER] 提示：ntheta>0 但未提供 theta0，θ 初猜将全部取 0。")
        self.resolve_quad_mode()

    # 供 AD tape 构建用：返回 a_double 初值
    def ad_initial_state(self, atheta):
        if callable(self._x0_from_theta_ad):
            return self._x0_from_theta_ad(atheta, self)
        return np.array([cppad_py.a_double(v) for v in self.x0], dtype=object)

    # 供数值 rollout / 可视化用：返回 float 初值
    # 触发条件必须与 ad_initial_state 完全一致（只看钩子是否可调用），否则
    # X_opt 会和 tape 内部真正优化的那条轨迹对不上。钩子需自行处理 theta=None。
    def numeric_initial_state(self, theta=None):
        if callable(self._x0_from_theta_numeric):
            return self._x0_from_theta_numeric(theta, self)
        return self.x0.copy()

    # ---------- 控制初猜展开 ----------
    def _expand_u0(self) -> np.ndarray:
        """
        把 self.u0 规范化成长度 N*nu 的扁平向量。
        支持形状：
          - 标量：同一个值填满所有步与分量
          - (nu,)：每步都用同一控制向量
          - (N,)：仅当 nu==1 有效，逐步给标量控制
          - (N, nu)：逐步控制矩阵
          - (N*nu,)：已扁平
        若 self.u0 为空，则优先用 bounds 的中点；否则用 0。
        """
        N, nu = self.N, self.nu
        if nu == 0:
            return np.empty(0, dtype=float)

        # 未提供 u0：尝试用 box bounds 的中点
        if self.u0 is None:
            bnds = None
            if callable(self.control_bounds_builder):
                try:
                    bnds = self.control_bounds_builder(self)
                except Exception:
                    bnds = None

            if isinstance(bnds, (list, tuple)) and len(bnds) == N * nu:
                mids = []
                for (lo, hi) in bnds:
                    lo = -np.inf if lo is None else lo
                    hi =  np.inf if hi is None else hi
                    if np.isfinite(lo) and np.isfinite(hi):
                        mids.append(0.5 * (lo + hi))
                    elif np.isfinite(lo) and not np.isfinite(hi):
                        # 只有下界：取 max(lo, 0)
                        mids.append(max(lo, 0.0))
                    elif not np.isfinite(lo) and np.isfinite(hi):
                        # 只有上界：取 min(hi, 0)
                        mids.append(min(hi, 0.0))
                    else:
                        # 无界：取 0
                        mids.append(0.0)
                return np.asarray(mids, dtype=float)

            # 没有 bounds 或不合法：回退全 0
            return np.zeros(N * nu, dtype=float)

        # 提供了 u0：规范各种形状
        u0 = np.asarray(self.u0, dtype=float)
        if u0.ndim == 0:
            # 标量
            return np.full(N * nu, float(u0), dtype=float)
        if u0.shape == (nu,):
            # 每步相同的控制向量
            return np.tile(u0, N).astype(float)
        if u0.shape == (N,):
            # 仅当 nu==1 合法
            if nu != 1:
                raise ValueError("u0 shape (N,) is only valid when nu == 1.")
            return u0.astype(float)
        if u0.shape == (N, nu):
            return u0.reshape(N * nu).astype(float)
        if u0.shape == (N * nu,):
            return u0.astype(float)

        raise ValueError(
            f"Unsupported u0 shape {u0.shape}; expected scalar, (nu,), (N,), (N,nu), or (N*nu,)."
        )

    # ---------- 决策变量初猜（控制 + 参数） ----------
    def initial_guess(self) -> np.ndarray:
        z = []

        # 控制初猜
        U0 = self._expand_u0()
        z.extend(U0.tolist())

        # 参数初猜
        if self.has_params():
            if self.theta0 is None:
                z.extend([0.0] * self.ntheta)
            else:
                z.extend(np.asarray(self.theta0, dtype=float).tolist())

        return np.asarray(z, dtype=float)

    # ---------- 盒约束集合 ----------
    def make_bounds(self):
        bounds = []
        if self.nu > 0:
            if callable(self.control_bounds_builder):
                bounds.extend(self.control_bounds_builder(self))
            else:
                # 默认无界
                bounds.extend([(-np.inf, np.inf)] * (self.N * self.nu))

        if self.has_params():
            if callable(self.param_bounds_builder):
                bounds.extend(self.param_bounds_builder(self))
            else:
                bounds.extend([(-np.inf, np.inf)] * self.ntheta)

        return bounds

    # ---------- 拆分决策向量 ----------
    def split_decision(self, z):
        z = np.asarray(z, dtype=float)
        U = z[: self.N * self.nu]
        theta = None
        if self.has_params():
            theta = z[self.N * self.nu : self.N * self.nu + self.ntheta]
        return U, theta
