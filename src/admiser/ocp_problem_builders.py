# ocp_problem_builders.py
from contextlib import contextmanager

import numpy as np
import cppad_py

from .ocp_integrators_utils import validate_quad_scheme

#: Package-wide default quadrature scheme: 4th order, matching the state
#: integrator, and costing no extra dynamics evaluations.
DEFAULT_QUAD_SCHEME = 'rk4'


class OCPProblem:
    """
    Unified container for an optimal control problem.

    - The objective is built by objective_builder(U[a_double], theta[a_double], problem).
    - Constraints are registered through the add_* API in canonical form and
      assembled into AD tapes together:
        equality  : G(z) = 0
        inequality: C(z) >= 0
    - Dynamics and integration are handled by `integrator`, which must accept
      accumulate_cb, t0 and quad.
    - Control initial guess u0 accepts several shapes and is expanded to a flat
      vector of length N*nu:
        * scalar : one value for every step and every control component
        * (nu,)  : the same control vector at every step
        * (N,)   : only valid when nu == 1 (one scalar control per step)
        * (N, nu): per-step control matrix
        * (N*nu,): already flat
      If u0 is not given, the midpoint of the control box bounds is used;
      unbounded components fall back to 0.
    """

    def __init__(
        self,
        *,
        N: int,
        dt: float,
        x0: np.ndarray,
        u0=None,                           # optional control initial guess (shapes above)
        dyn=None,                          # dyn(x, u, theta=None) -> np.array(dtype=object)
        integrator=None,                   # integrator(x, u, dt_k, f, accumulate_cb=None, t0=None, quad='rk4')
        nu: int,
        nx: int,
        objective_builder=None,            # required
        constraint_builder=None,           # legacy interface, may be None (not recommended)
        control_bounds_builder=None,       # -> list[(low, high)] of length N*nu
        ntheta: int = 0,
        theta0: np.ndarray | None = None,
        param_bounds_builder=None,         # -> list[(low, high)] of length ntheta
        x0_from_theta_ad=None,             # optional: a_double version of x0(theta)
        x0_from_theta_numeric=None,        # optional: numeric version of x0(theta)
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
        # Legacy hook kept for backwards compatibility; prefer the add_* API.
        self.constraint_builder = constraint_builder

        self.control_bounds_builder = control_bounds_builder

        self.ntheta = int(ntheta) if ntheta is not None else 0
        self.theta0 = None if theta0 is None else np.asarray(theta0, dtype=float)
        self.param_bounds_builder = param_bounds_builder

        # Optional theta-dependent initial state
        self._x0_from_theta_ad      = x0_from_theta_ad
        self._x0_from_theta_numeric = x0_from_theta_numeric

        # Canonical constraint registries
        self.term_eq_specs   = []  # psi_i(xT, theta) = 0
        self.term_ineq_specs = []  # phi_i(xT, theta) <= 0 / >= 0
        self.int_eq_specs    = []  # int q_i dt  = b_i
        self.int_ineq_specs  = []  # int q_i dt <=/>= b_i
        self.path_eq_specs   = []  # path equality h(t)=0 -> int h^2 dt = 0, or int smooth|h| dt = 0
        self.path_ineq_specs = []  # path inequality h<=0 -> int L_eps(h) dt <= gamma

        # Substep quadrature scheme; see ocp_integrators_utils.QUAD_SCHEMES.
        # None means "unspecified", in which case the quad carried by
        # objective_builder decides (see resolve_quad_scheme). Setting it
        # explicitly overrides that and applies to the objective and all
        # constraints alike.
        self.quad_scheme = None

        # Solve strategy for the path-constraint transcription; see set_transcription().
        self.transcription = dict(mode="single", n_rounds=1, shrink=0.1)

        # Transient epsilon adjustment. Only set briefly by OCPSolver through
        # scaled_eps() and always restored afterwards, so the value registered
        # in spec["eps"] always stays exactly what the user wrote.
        self._eps_factor = 1.0
        self._eps_override = None

    # ---------- transcription strategy ----------
    def set_transcription(self, mode: str = "single", n_rounds: int = 4,
                          shrink: float = 0.1) -> "OCPProblem":
        """
        Declare how the path-inequality transcription should be solved. This
        belongs to the problem definition, so the solving side never has to
        branch on it.

        mode="single" (default)
            Solve once with the eps/gamma registered by add_path_ineq.

        mode="continuation"
            Solve by an eps -> 0 continuation. The eps registered with
            add_path_ineq is treated as the FINAL value; the rounds actually run
                eps/shrink^(n_rounds-1), ..., eps/shrink, eps
            each warm-started from the previous round's solution, so the last
            round lands exactly on the registered value. For constraints whose
            gamma was left automatic (add_path_ineq called without gamma), gamma
            is re-derived as T*eps/4 every round.

            The starting point is given as a shrink RATIO rather than an absolute
            value: eps carries the units of its own h, so several path constraints
            cannot share one absolute start, but they can share a ratio.

        Why continuation: with a large eps, L_eps is smooth and easy to solve but
        the constraint is loose; with a small eps it approaches the true
        constraint but also approaches a nondifferentiable hinge, where a cold
        start easily stalls on the boundary. Shrinking gradually gets both.
        """
        if mode not in ("single", "continuation"):
            raise ValueError(f"mode must be 'single' or 'continuation', got {mode!r}")
        if mode == "continuation":
            if int(n_rounds) < 1:
                raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
            if not (0.0 < float(shrink) < 1.0):
                raise ValueError(f"shrink must lie in (0, 1), got {shrink}")
            if not self.path_ineq_specs:
                raise ValueError(
                    "No path inequality is registered (add_path_ineq), so there is "
                    "nothing to continue on; use set_transcription(mode='single') "
                    "or simply do not call this method."
                )
        self.transcription = dict(mode=mode,
                                  n_rounds=1 if mode == "single" else int(n_rounds),
                                  shrink=float(shrink))
        return self

    def eps_factors(self) -> list:
        """
        Multipliers applied to the registered eps, one per round, in DESCENDING
        order; the last one is always 1.0 (the registered value itself).

        "single" returns [1.0]; "continuation" returns [(1/s)^(n-1), ..., 1/s, 1.0].
        Example: n_rounds=4, shrink=0.1, registered eps=1e-4 -> the rounds use
        eps = 1e-1, 1e-2, 1e-3, 1e-4.
        """
        cfg = self.transcription
        if cfg["mode"] == "single":
            return [1.0]
        n, s = cfg["n_rounds"], cfg["shrink"]
        return [(1.0 / s) ** (n - 1 - k) for k in range(n)]

    @contextmanager
    def scaled_eps(self, factor: float = 1.0, override: float | None = None):
        """
        Temporarily change the effective eps of every path constraint inside the
        with-block: `override` wins (absolute value), otherwise the registered
        value is scaled by `factor`. The previous state is always restored on
        exit, so spec["eps"] is never mutated and the same problem can be solved
        repeatedly without drifting.
        """
        old = (self._eps_factor, self._eps_override)
        self._eps_factor = float(factor)
        self._eps_override = None if override is None else float(override)
        try:
            yield self
        finally:
            self._eps_factor, self._eps_override = old

    def effective_eps(self, spec: dict) -> float:
        """The eps currently in effect for this path constraint."""
        if self._eps_override is not None:
            return self._eps_override
        return float(spec["eps"]) * self._eps_factor

    def effective_gamma(self, spec: dict) -> float:
        """
        The gamma currently in effect: automatic ones follow eps, explicitly
        given ones stay fixed.
        """
        if spec.get("gamma_auto", False):
            return self.auto_gamma(self.effective_eps(spec))
        return float(spec["gamma"])

    # ---------- quadrature scheme resolution ----------
    def resolve_quad_scheme(self) -> str:
        """
        The single place where the quadrature scheme is resolved. The objective
        and every constraint must go through it. Priority:
          1. problem.quad_scheme (set explicitly)
          2. the quad recorded on objective_builder by make_builders(quad=...)
          3. the package default DEFAULT_QUAD_SCHEME

        An unknown name raises here rather than deep inside the integrator.
        """
        scheme = getattr(self, "quad_scheme", None)
        if scheme is None:
            scheme = getattr(self.objective_builder, "quad", None)
        if scheme is None:
            scheme = DEFAULT_QUAD_SCHEME
        return validate_quad_scheme(scheme)

    # ---------- canonical constraint registration ----------
    def add_terminal_eq(self, psi):
        """psi(xT, theta) -> a_double, or a vector of a_double."""
        self.term_eq_specs.append(dict(psi=psi))

    def add_terminal_ineq(self, phi, sense: str = "<="):
        """phi(xT, theta) <= 0 or >= 0; sense in {'<=', '>='}."""
        assert sense in ("<=", ">=")
        self.term_ineq_specs.append(dict(phi=phi, sense=sense))

    def add_integral_eq(self, qfun, target: float):
        """int q dt = target, with qfun(t, x, u, theta) -> a_double."""
        self.int_eq_specs.append(dict(qfun=qfun, target=float(target)))

    def add_integral_ineq(self, qfun, bound: float, sense: str = "<="):
        """int q dt <=/>= bound; sense in {'<=', '>='}."""
        assert sense in ("<=", ">=")
        self.int_ineq_specs.append(dict(qfun=qfun, bound=float(bound), sense=sense))

    def auto_gamma(self, eps: float) -> float:
        """
        The gamma that is consistent with a given eps in the constraint
        transcription: gamma = T*eps/4, with T = N*dt.

        Rationale: L_eps(h) - max(h, 0) lies in [0, eps/4], and on h <= 0 the
        maximum of L_eps is attained exactly at L_eps(0) = eps/4. Therefore

          * any trajectory that genuinely satisfies h(t) <= 0 also satisfies
            int L_eps dt <= T*eps/4, so the transcribed problem does not exclude
            the feasible set (including the optimum) of the original problem;
          * conversely int max(h, 0) dt <= int L_eps dt <= gamma = T*eps/4, so the
            L1 norm of the violation is bounded by T*eps/4 and vanishes as eps -> 0.

        Taking gamma = 0 forces L_eps == 0, which is equivalent to h(t) <= -eps
        (a strictly interior solution). That is both overly conservative and a
        frequent cause of SLSQP losing its descent direction on the boundary.
        """
        return float(self.N * self.dt * float(eps) / 4.0)

    def add_path_ineq(self, hfun, eps: float, gamma: float | None = None):
        """
        h(x, u, theta) <= 0   -->   int L_eps(h) dt <= gamma

        gamma=None (default) means gamma is taken automatically as
        auto_gamma(eps) = T*eps/4, and is kept in sync whenever a continuation
        solve shrinks eps. Passing an explicit number pins it instead.
        """
        auto = gamma is None
        self.path_ineq_specs.append(dict(
            hfun=hfun, eps=float(eps),
            gamma=self.auto_gamma(eps) if auto else float(gamma),
            gamma_auto=auto,
        ))

    def add_path_eq(self, hfun, mode: str = "L2", eps_abs: float = 1e-6):
        """
        Path equality h(t) = 0:
        - mode='L2' : int h^2 dt = 0 (recommended)
        - mode='abs': int smooth_abs_eps(h) dt = 0
        """
        assert mode in ("L2", "abs")
        self.path_eq_specs.append(dict(hfun=hfun, mode=mode, eps_abs=float(eps_abs)))

    # ---------- basic utilities ----------
    def has_params(self) -> bool:
        return self.ntheta > 0

    def dt_of_segment(self, k: int):
        """
        Duration of segment k, as an a_double. Both the objective tape and the
        constraint tapes must go through this single entry point; otherwise a
        future time-scaling transform (CPET) would silently give them different
        time grids. Subclasses or users may implement _current_dt(k) returning an
        a_double to support variable step sizes.
        """
        if callable(getattr(self, "_current_dt", None)):
            return self._current_dt(k)
        return cppad_py.a_double(self.dt)

    def validate(self) -> None:
        """
        Structural check run before taping, so that CppAD/SciPy internals are
        never the first thing a user sees when something is misconfigured.
        """
        if not callable(self.objective_builder):
            raise ValueError(
                "OCPProblem.objective_builder is not set. Build one with "
                "make_builders(dyn=..., L=..., Phi=...) and pass it as "
                "OCPProblem(objective_builder=...)."
            )
        if not callable(self.integrator):
            raise ValueError(
                "OCPProblem.integrator is not set, e.g. partial(rk4_substeps, m_sub=10)."
            )
        if not callable(self.dyn):
            raise ValueError("OCPProblem.dyn is not set.")
        if self.N <= 0 or self.dt <= 0:
            raise ValueError(f"N > 0 and dt > 0 are required, got N={self.N}, dt={self.dt}.")
        if self.x0.shape != (self.nx,):
            raise ValueError(f"x0 has shape {self.x0.shape}, which does not match nx={self.nx}.")

        n_z = self.N * self.nu + (self.ntheta if self.has_params() else 0)
        bnds = self.make_bounds()
        if len(bnds) != n_z:
            raise ValueError(
                f"Got {len(bnds)} box bounds but {n_z} decision variables (= N*nu + ntheta); "
                "check the lengths returned by control_bounds_builder / param_bounds_builder."
            )
        if self.initial_guess().size != n_z:
            raise ValueError(
                f"Initial guess has size {self.initial_guess().size} but there are {n_z} "
                "decision variables (check u0 / theta0)."
            )
        if self.has_params() and self.theta0 is None:
            print("[ADMISER] Note: ntheta > 0 but no theta0 was given; theta starts at all zeros.")
        self.resolve_quad_scheme()

    # Initial state for AD taping: returns a_double values.
    def ad_initial_state(self, atheta):
        if callable(self._x0_from_theta_ad):
            return self._x0_from_theta_ad(atheta, self)
        return np.array([cppad_py.a_double(v) for v in self.x0], dtype=object)

    # Initial state for the numeric rollout / plotting: returns float values.
    # The trigger condition must match ad_initial_state exactly (only whether the
    # hook is callable); otherwise X_opt would not correspond to the trajectory
    # that was actually optimised inside the tape. The hook must handle theta=None.
    def numeric_initial_state(self, theta=None):
        if callable(self._x0_from_theta_numeric):
            return self._x0_from_theta_numeric(theta, self)
        return self.x0.copy()

    # ---------- control initial guess expansion ----------
    def _expand_u0(self) -> np.ndarray:
        """
        Normalise self.u0 into a flat vector of length N*nu.

        Supported shapes:
          - scalar : one value fills every step and component
          - (nu,)  : the same control vector at every step
          - (N,)   : only valid when nu == 1, one scalar control per step
          - (N, nu): per-step control matrix
          - (N*nu,): already flat

        If self.u0 is None, the midpoint of the box bounds is used, else zeros.
        """
        N, nu = self.N, self.nu
        if nu == 0:
            return np.empty(0, dtype=float)

        # No u0 given: try the midpoint of the box bounds.
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
                        # lower bound only: take max(lo, 0)
                        mids.append(max(lo, 0.0))
                    elif not np.isfinite(lo) and np.isfinite(hi):
                        # upper bound only: take min(hi, 0)
                        mids.append(min(hi, 0.0))
                    else:
                        # unbounded: take 0
                        mids.append(0.0)
                return np.asarray(mids, dtype=float)

            # No usable bounds: fall back to all zeros.
            return np.zeros(N * nu, dtype=float)

        # u0 given: normalise the accepted shapes.
        u0 = np.asarray(self.u0, dtype=float)
        if u0.ndim == 0:
            # scalar
            return np.full(N * nu, float(u0), dtype=float)
        if u0.shape == (nu,):
            # same control vector at every step
            return np.tile(u0, N).astype(float)
        if u0.shape == (N,):
            # only valid when nu == 1
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

    # ---------- decision-variable initial guess (controls + parameters) ----------
    def initial_guess(self) -> np.ndarray:
        z = []

        # controls
        U0 = self._expand_u0()
        z.extend(U0.tolist())

        # system parameters
        if self.has_params():
            if self.theta0 is None:
                z.extend([0.0] * self.ntheta)
            else:
                z.extend(np.asarray(self.theta0, dtype=float).tolist())

        return np.asarray(z, dtype=float)

    # ---------- box bounds ----------
    def make_bounds(self):
        bounds = []
        if self.nu > 0:
            if callable(self.control_bounds_builder):
                bounds.extend(self.control_bounds_builder(self))
            else:
                # unbounded by default
                bounds.extend([(-np.inf, np.inf)] * (self.N * self.nu))

        if self.has_params():
            if callable(self.param_bounds_builder):
                bounds.extend(self.param_bounds_builder(self))
            else:
                bounds.extend([(-np.inf, np.inf)] * self.ntheta)

        return bounds

    # ---------- split the decision vector ----------
    def split_decision(self, z):
        z = np.asarray(z, dtype=float)
        U = z[: self.N * self.nu]
        theta = None
        if self.has_params():
            theta = z[self.N * self.nu : self.N * self.nu + self.ntheta]
        return U, theta
