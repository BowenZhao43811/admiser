# problem_definition.py
import numpy as np
import jax.numpy as jnp

from .quadrature import validate_quad_scheme

#: Package-wide default quadrature scheme: 4th order, matching the state
#: integrator, and costing no extra dynamics evaluations.
DEFAULT_QUAD_SCHEME = 'rk4'

#: Default number of RK4 substeps per control segment.
DEFAULT_M_SUB = 5


class OCPProblem:
    """
    Unified container for an optimal control problem.

    - The objective is declared with make_builders(dyn=..., L=..., Phi=...).
    - Constraints are registered through the add_* API in canonical form:
        equality  : G(z) = 0
        inequality: C(z) >= 0
    - The state is advanced by classical RK4, with m_sub substeps on every
      control segment.
    - Every model function -- dyn, L, Phi and each constraint function -- must be
      written with jax.numpy (jnp.array, jnp.exp, jnp.where, ...), because the
      solver differentiates them with JAX.
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
        dyn=None,                          # dyn(x, u, theta=None) -> jnp.array of length nx
        m_sub: int = DEFAULT_M_SUB,        # RK4 substeps per control segment
        nu: int,
        nx: int,
        objective_builder=None,            # required: make_builders(dyn=..., L=..., Phi=...)
        control_bounds_builder=None,       # -> list[(low, high)] of length N*nu
        ntheta: int = 0,
        theta0: np.ndarray | None = None,
        param_bounds_builder=None,         # -> list[(low, high)] of length ntheta
        x0_from_theta=None,                # optional: x0_from_theta(theta, problem) -> x(0)
    ):
        self.N  = int(N)
        self.dt = float(dt)
        self.x0 = np.asarray(x0, dtype=float)
        self.u0 = None if u0 is None else np.asarray(u0, dtype=float)

        self.nu = int(nu)
        self.nx = int(nx)

        self.dyn = dyn
        # The state is advanced by RK4 with this many substeps per segment. A larger
        # m_sub means a more accurate state and more accurate integrals, at a cost
        # that grows linearly with it.
        self.m_sub = int(m_sub)

        self.objective_builder = objective_builder

        self.control_bounds_builder = control_bounds_builder

        self.ntheta = int(ntheta) if ntheta is not None else 0
        self.theta0 = None if theta0 is None else np.asarray(theta0, dtype=float)
        self.param_bounds_builder = param_bounds_builder

        # Optional theta-dependent initial state; see initial_state().
        self.x0_from_theta = x0_from_theta

        # Canonical constraint registries
        self.term_eq_specs   = []  # psi_i(xT, theta) = 0
        self.term_ineq_specs = []  # phi_i(xT, theta) <= 0 / >= 0
        self.int_eq_specs    = []  # int q_i dt  = b_i
        self.int_ineq_specs  = []  # int q_i dt <=/>= b_i
        self.path_eq_specs   = []  # path equality h(t)=0 -> int h^2 dt = 0, or int smooth|h| dt = 0
        self.path_ineq_specs = []  # path inequality h<=0 -> int L_eps(h) dt <= gamma

        # Substep quadrature scheme; see quadrature.QUAD_SCHEMES.
        # None means "unspecified", in which case the quad carried by
        # objective_builder decides (see resolve_quad_scheme). Setting it
        # explicitly overrides that and applies to the objective and all
        # constraints alike.
        self.quad_scheme = None

        # Solve strategy for the path-constraint transcription; see set_transcription().
        self.transcription = dict(mode="single", n_rounds=1, shrink=0.1)

        # Automatic problem scaling; see set_scaling() and problem_scaling.
        # On by default, because the failure it prevents is silent: SLSQP's ftol is
        # an absolute test, so a badly scaled objective makes it stop early at a
        # point that may not even be feasible.
        self.scaling = dict(objective="auto", constraints="auto")

        # Time-scaling transform (CPET); None until set_time_scaling() is called.
        # See that method for what it does and why.
        self.time_scaling = None

    # ---------- automatic scaling ----------
    def set_scaling(self, objective="auto", constraints="auto") -> "OCPProblem":
        """
        Choose how the problem is rescaled before it is handed to SLSQP.

        Both are "auto" by default. Write your objective and constraints in
        whatever units are natural for your problem; the solver normalises them
        internally and converts everything it reports back into your units.

        objective : "auto"  normalise |J| to about 1, estimated from a few fixed
                            probe points around the initial guess
                    "none"  leave it alone
                    a positive number to set the factor yourself
        constraints : "auto" scale each constraint ROW by 1/||grad g_i||
                      "none" leave them alone

        Why this is on by default
        -------------------------
        SLSQP's ftol compares the ABSOLUTE change in the objective. With an
        objective of order 1e6, ftol=1e-9 demands 1e-15 relative accuracy, below
        machine precision, and the line search gives up after a few iterations --
        at a point that may be infeasible, with no error raised. Normalising turns
        that absolute test into an effectively relative one.

        The two are scaled by different rules, for different reasons; see the
        module docstring of problem_scaling for the details.
        """
        if constraints not in ("auto", "none"):
            raise ValueError(
                f"constraints scaling must be 'auto' or 'none', got {constraints!r}")
        if not (objective in ("auto", "none")
                or (isinstance(objective, (int, float)) and not isinstance(objective, bool)
                    and objective > 0.0)):
            raise ValueError(
                f"objective scaling must be 'auto', 'none' or a positive number, "
                f"got {objective!r}")
        self.scaling = dict(objective=objective, constraints=constraints)
        return self

    # ---------- time-scaling transform (CPET) ----------
    def set_time_scaling(self, tau0=None, tau_min=0.0, tau_max=None,
                         total_time=None) -> "OCPProblem":
        """
        Turn the segment durations into decision variables (Teo's Control
        Parameterization Enhancing Transform).

        What it does
        ------------
        Without it the knot points are fixed and uniform: the optimiser chooses
        only the control VALUE on each segment, never where the segments end. For
        a bang-bang solution the switching instant can then only land on a grid
        point, so resolving it costs a fine grid -- and the grid is uniform, so
        the cost is paid everywhere, not just near the switch.

        CPET introduces a new independent variable s on which the grid IS uniform,
        and lets real time flow at an optimisable rate:

            dt/ds = tau_k    for s in [k-1, k)

        Segment k, of width 1 in s, therefore has duration tau_k in real time. The
        augmented system is

            dx/ds = tau_k * f(t, x, u_k),      dt/ds = tau_k

        so the ODE is still solved on a fixed uniform grid, while all the awkward
        variability has been absorbed into ordinary decision variables tau_k.

        The decision vector becomes  z = [U (N*nu) ; theta (ntheta) ; tau (N)].

        Free terminal time comes out for free: leave total_time as None and make
        the objective the elapsed time, L(t, x, u, theta) = 1.0, since
        int 1 dt = sum(tau). That replaces the usual trick of hand-adding an extra
        state and control to carry the horizon.

        Parameters
        ----------
        tau0 : initial guess for the durations. Scalar, array of length N, or None
               for a uniform split, tau_k = dt, whose sum is the nominal N*dt.
        tau_min, tau_max : box bounds on every duration. tau_min defaults to 0,
               which is the standard choice: a duration may collapse to zero, and
               that is precisely how the optimiser drops a segment it does not
               need. Negative durations would run time backwards and are never
               allowed. tau_max=None means unbounded above.
        total_time : if given, pin the horizon by registering sum(tau) = total_time.
               That constraint is nothing special -- it is exactly
                   add_integral_eq(qfun=lambda t, x, u, th: 1.0, target=total_time)
               because int 1 dt over the whole horizon is sum(tau). Leave it None
               for a free horizon.
        """
        if tau0 is None:
            tau = np.full(self.N, self.dt, dtype=float)
        else:
            tau = np.asarray(tau0, dtype=float)
            if tau.ndim == 0:
                tau = np.full(self.N, float(tau), dtype=float)
            if tau.shape != (self.N,):
                raise ValueError(
                    f"tau0 must be a scalar or have shape ({self.N},), got {tau.shape}")
        if np.any(tau < 0.0):
            raise ValueError("tau0 must be non-negative; durations cannot run backwards")

        lo = float(tau_min)
        if lo < 0.0:
            raise ValueError(f"tau_min must be >= 0, got {tau_min}")
        hi = np.inf if tau_max is None else float(tau_max)
        if hi <= lo:
            raise ValueError(f"tau_max must exceed tau_min, got {tau_max} <= {tau_min}")

        self.time_scaling = dict(tau0=tau, tau_min=lo, tau_max=hi,
                                 total_time=None if total_time is None else float(total_time))

        if total_time is not None:
            # int 1 dt == sum(tau), so a plain integral equality pins the horizon.
            self.add_integral_eq(qfun=lambda t, x, u, th: 1.0, target=float(total_time))
        return self

    def has_time_scaling(self) -> bool:
        return self.time_scaling is not None

    @property
    def n_tau(self) -> int:
        """Number of duration variables: N when CPET is on, otherwise 0."""
        return self.N if self.has_time_scaling() else 0

    def nominal_horizon(self) -> float:
        """
        The problem's total duration, used wherever a horizon length is needed
        (auto_gamma, for instance).

        Without CPET that is simply N*dt. With CPET the horizon is a variable, so
        the pinned total_time is used when there is one, and otherwise the sum of
        the initial durations -- a nominal figure, which is all auto_gamma needs.
        """
        if not self.has_time_scaling():
            return float(self.N * self.dt)
        if self.time_scaling["total_time"] is not None:
            return float(self.time_scaling["total_time"])
        return float(np.sum(self.time_scaling["tau0"]))

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

    def eps_gamma(self, factor: float = 1.0, eps: float | None = None):
        """
        The eps and gamma of every path inequality for one round, as two arrays
        with one entry per add_path_ineq call, in registration order.

        eps   : the registered eps times `factor`; an explicit `eps` replaces it
                for every path constraint
        gamma : automatic ones follow eps (auto_gamma); explicitly given ones stay
                as registered

        The solver passes these arrays into the compiled NLP at call time, so the
        registered values themselves are never modified.
        """
        specs = self.path_ineq_specs
        if eps is None:
            eps_arr = np.array([float(s["eps"]) * float(factor) for s in specs], dtype=float)
        else:
            eps_arr = np.full(len(specs), float(eps), dtype=float)
        gamma_arr = np.array(
            [self.auto_gamma(e) if s.get("gamma_auto", False) else float(s["gamma"])
             for s, e in zip(specs, eps_arr)],
            dtype=float)
        return eps_arr, gamma_arr

    def transcription_rounds(self) -> list:
        """
        The (eps, gamma) arrays of every round, in the order the rounds run;
        see eps_factors() and eps_gamma().
        """
        return [self.eps_gamma(factor) for factor in self.eps_factors()]

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
        """psi(xT, theta) = 0, with psi returning a scalar or a 1-D array."""
        self.term_eq_specs.append(dict(psi=psi))

    def add_terminal_ineq(self, phi, sense: str = "<="):
        """phi(xT, theta) <= 0 or >= 0; sense in {'<=', '>='}."""
        assert sense in ("<=", ">=")
        self.term_ineq_specs.append(dict(phi=phi, sense=sense))

    def add_integral_eq(self, qfun, target: float):
        """int q dt = target, with qfun(t, x, u, theta) -> a scalar."""
        self.int_eq_specs.append(dict(qfun=qfun, target=float(target)))

    def add_integral_ineq(self, qfun, bound: float, sense: str = "<="):
        """int q dt <=/>= bound; sense in {'<=', '>='}."""
        assert sense in ("<=", ">=")
        self.int_ineq_specs.append(dict(qfun=qfun, bound=float(bound), sense=sense))

    def auto_gamma(self, eps: float) -> float:
        """
        The gamma that is consistent with a given eps in the constraint
        transcription: gamma = T*eps/4, with T the horizon (N*dt normally; see
        nominal_horizon(), since CPET turns the horizon into a variable).

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
        return float(self.nominal_horizon() * float(eps) / 4.0)

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

    def validate(self) -> None:
        """
        Structural check run before anything is compiled, so that JAX/SciPy
        internals are never the first thing a user sees when something is
        misconfigured.
        """
        ob = self.objective_builder
        if not all(hasattr(ob, name) for name in ("dyn", "L", "Phi", "quad")):
            raise ValueError(
                "OCPProblem.objective_builder must be created with "
                "make_builders(dyn=..., L=..., Phi=...) and passed as "
                "OCPProblem(objective_builder=...)."
            )
        if not callable(self.dyn):
            raise ValueError("OCPProblem.dyn is not set.")
        if ob.dyn is not self.dyn:
            # There is exactly one rollout, driven by problem.dyn. A different dyn
            # in make_builders would silently be ignored, so refuse it instead.
            raise ValueError(
                "make_builders(dyn=...) and OCPProblem(dyn=...) were given different "
                "functions; pass the same dynamics to both."
            )
        if self.m_sub < 1:
            raise ValueError(f"m_sub must be at least 1, got {self.m_sub}.")
        if self.N <= 0 or self.dt <= 0:
            raise ValueError(f"N > 0 and dt > 0 are required, got N={self.N}, dt={self.dt}.")
        if self.x0.shape != (self.nx,):
            raise ValueError(f"x0 has shape {self.x0.shape}, which does not match nx={self.nx}.")

        n_z = (self.N * self.nu
               + (self.ntheta if self.has_params() else 0)
               + self.n_tau)
        bnds = self.make_bounds()
        if len(bnds) != n_z:
            raise ValueError(
                f"Got {len(bnds)} box bounds but {n_z} decision variables "
                f"(= N*nu + ntheta + n_tau = {self.N * self.nu} + "
                f"{self.ntheta if self.has_params() else 0} + {self.n_tau}); "
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

    def initial_state(self, theta=None):
        """
        The initial state x(0): x0_from_theta(theta, problem) when that hook was
        given, otherwise the fixed x0.

        JAX evaluates and differentiates the same function, so one hook serves
        both the optimisation and the replay of the optimal trajectory. (The CppAD
        version needed two: one for AD values and one for plain floats.) theta is
        None when the problem has no system parameters.
        """
        if callable(self.x0_from_theta):
            return jnp.asarray(self.x0_from_theta(theta, self), dtype=float)
        return jnp.asarray(self.x0, dtype=float)

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

        # segment durations, when CPET is enabled
        if self.has_time_scaling():
            z.extend(self.time_scaling["tau0"].tolist())

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

        if self.has_time_scaling():
            ts = self.time_scaling
            bounds.extend([(ts["tau_min"], ts["tau_max"])] * self.N)

        return bounds

    # ---------- split the decision vector ----------
    def split_decision(self, z):
        """
        Split z into (U, theta, tau), following the layout
            z = [U (N*nu) ; theta (ntheta) ; tau (N)]
        theta is None when the problem has no system parameters, and tau is None
        when the time-scaling transform is off.
        """
        z = np.asarray(z, dtype=float)
        i = self.N * self.nu
        U = z[:i]

        theta = None
        if self.has_params():
            theta = z[i: i + self.ntheta]
            i += self.ntheta

        tau = None
        if self.has_time_scaling():
            tau = z[i: i + self.N]

        return U, theta, tau

    def segment_durations(self, tau=None):
        """
        The duration of each segment as a float array of length N: the optimised
        tau under CPET, or a uniform dt otherwise. Used by the numeric rollout and
        for building the time grid.
        """
        if tau is not None:
            return np.asarray(tau, dtype=float)
        if self.has_time_scaling():
            return np.asarray(self.time_scaling["tau0"], dtype=float)
        return np.full(self.N, self.dt, dtype=float)
