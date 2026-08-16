# sqp_solver.py

import inspect

import numpy as np
from scipy.optimize import minimize

from .ad_tape import build_ad_tape
from .nlp_functions import NLPFunctions
from .problem_scaling import compute_scaling


def _bind_dyn_numeric(dyn, theta):
    """
    Wrap dyn into f(x, u) for the numeric path. It must detect 2-argument versus
    3-argument signatures by exactly the same rule as _bind_dyn_with_theta_ad on
    the AD side; otherwise a problem whose dyn only takes (x, u) would raise a
    TypeError only after the whole optimisation has finished, while replaying the
    trajectory.
    """
    n_params = len(inspect.signature(dyn).parameters)
    if n_params == 2:
        return lambda x, u: np.asarray(dyn(x, u), dtype=float)
    return lambda x, u: np.asarray(dyn(x, u, theta), dtype=float)


class OCPSolver:
    """
    General optimal control solver.

    ------------- only two public methods -------------
    solve(...)   Solve. Whether that is a single solve or an eps -> 0 continuation
                 is decided by the problem itself, declared with
                 OCPProblem.set_transcription(); the solving side never branches on it.
    to_nlp(...)  Only transcribe the OCP into an NLP (record the AD tape) and return
                 an object that evaluates it and its derivatives -- NO optimisation.
                 Useful for checking gradients, e.g. against finite differences.

    ------------- result dictionary -------------
    scipy_result : the SciPy result of the final round
    U_opt        : ndarray (N*nu,), optimal control
    theta_opt    : ndarray (ntheta,), optimal system parameters (None if there are none)
    tau_opt      : ndarray (N,), optimal segment durations under the time-scaling
                   transform (None when it is off)
    T_opt        : float, the optimised horizon sum(tau) (None when it is off)
    J_opt        : float, optimal cost
    t_opt        : ndarray (N+1,), time grid; non-uniform under the time-scaling transform
    X_opt        : ndarray (N+1, nx), optimal state trajectory
    eq_resid     : ndarray, equality residual G(z), should be ~ 0 (None if there are none)
    ineq_resid   : ndarray, inequality residual C(z), should be >= 0 (None if there are none)
    path_viol    : ndarray, max h(t) on the grid for each path inequality, should be <= 0
                   (None if there are none)
    history_cost : ndarray, cost history of the final round's iterations
    scaling      : the ProblemScaling that was applied. Every value above is
                   already converted back into the user's units; this is here so
                   the transform is inspectable rather than invisible
    rounds       : list[dict], per round: eps, gamma, J_opt, max_path_viol, status, nit.
                   Length 1 in "single" mode, so callers never need to branch on the mode.

    ------------- notes -------------
    - solve() never mutates the problem: eps is only scaled temporarily during a
      round and restored afterwards, so the same problem can be solved repeatedly
      with identical results.
    """

    def __init__(self, problem: object):
        self.problem = problem
        self.taped = None      # the TapedNLP recorded by _build_tape()
        self.opt_fun = None    # SciPy-facing view of that tape
        # Scaling factors, computed once on the first tape and then FROZEN. They
        # must not be re-estimated per continuation round: each round would then
        # optimise a differently scaled problem, the warm start would lose its
        # meaning, and the per-round objectives would not be comparable.
        self.scaling = None
        self.history_cost = []

    # ================= public interface =================

    def solve(self, maxiter=5000, ftol=1e-8, disp=False, verbose=None, z0=None):
        """
        Solve the problem. The mode comes from problem.set_transcription():

          mode="single"       : solve once with the registered eps/gamma
          mode="continuation" : solve over a descending geometric eps schedule,
                                warm-starting each round from the previous solution

        Parameters
        ----------
        maxiter, ftol, disp : forwarded to SciPy SLSQP for every round
                              (disp prints SLSQP's own convergence report)
        verbose : whether to print eps/gamma/J/violation per round. None means
                  "print in continuation mode, stay quiet in single mode".
        z0      : starting point for the first round; defaults to problem.initial_guess()
        """
        p = self.problem
        factors = p.eps_factors()
        if verbose is None:
            verbose = len(factors) > 1

        z = p.initial_guess() if z0 is None else np.asarray(z0, dtype=float)
        rounds, res = [], None
        announced = False

        for k, factor in enumerate(factors):
            # eps is scaled only inside this with-block and restored on exit, so
            # spec["eps"] always keeps the value the user registered.
            with p.scaled_eps(factor=factor):
                self._build_tape()
                # Say once, up front, that the numbers were rescaled and by how
                # much. An automatic transform that changes the iteration history
                # must never be invisible.
                if verbose and not announced:
                    print(self.scaling.describe())
                    announced = True
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
                e = f"{min(eps_now):.3e}" if eps_now else "-"
                # max_viol is nan when the diagnostic could not be evaluated (a
                # hfun that only accepts AD types); say so rather than print nan.
                v = "n/a" if not np.isfinite(max_viol) else f"{max_viol:+.3e}"
                print(f"[ADMISER] round {k+1}/{len(factors)}  eps={e}  "
                      f"J={res['J_opt']:+.8g}  max h(t)={v}  "
                      f"status={res['scipy_result'].status}  nit={res['scipy_result'].nit}")

        res["rounds"] = rounds
        return res

    def to_nlp(self, eps=None):
        """
        Transcribe the OCP into an NLP and return an object that evaluates it --
        without running any optimisation.

        The returned object exposes objective_fun / objective_grad / eq_fun /
        eq_jac / ineq_fun / ineq_jac. The typical use is comparing the AD gradient
        against finite differences:

            nlp = OCPSolver(problem).to_nlp()
            g_ad = nlp.objective_grad(z)

        eps : which eps to tape with. None (default) uses the eps the FIRST
              continuation round would actually use (in "single" mode that is the
              registered value itself). Passing a number applies it to every path
              constraint.
        """
        p = self.problem
        factor = p.eps_factors()[0]
        with p.scaled_eps(factor=factor, override=eps):
            # Deliberately unscaled: this is meant to be the user's own problem,
            # so a gradient checked against finite differences here is a gradient
            # of what the user wrote, not of an internally rescaled version.
            self._build_tape(scaled=False)
        return self.opt_fun

    # ================= internals =================

    def _build_tape(self, scaled=True):
        """
        Record the AD tape. eps is baked into it as a constant, which is why every
        continuation round has to re-record.

        The tape itself is always in the user's units. When `scaled` is set, the
        SciPy-facing view multiplies by the frozen scaling factors on the way out;
        those factors are estimated once, from an UNSCALED view of the first tape,
        so they describe the problem as the user wrote it.
        """
        z0 = self.problem.initial_guess()
        self.taped = build_ad_tape(z0, self.problem)

        if not scaled:
            self.opt_fun = NLPFunctions(self.taped)
            return self.opt_fun

        if self.scaling is None:
            cfg = getattr(self.problem, "scaling", None) or {}
            unscaled = NLPFunctions(self.taped)
            self.scaling = compute_scaling(
                unscaled, z0, self.problem.make_bounds(),
                objective=cfg.get("objective", "auto"),
                constraints=cfg.get("constraints", "auto"),
            )
        self.opt_fun = NLPFunctions(self.taped, self.scaling)
        return self.opt_fun

    def _solve_once(self, z0, maxiter, ftol, disp):
        """Solve the NLP once using the tapes currently recorded."""
        bounds = self.problem.make_bounds()

        constraints = []
        if self.opt_fun.has_eq:
            constraints.append({'type': 'eq', 'fun': self.opt_fun.eq_fun, 'jac': self.opt_fun.eq_jac})
        if self.opt_fun.has_ineq:
            constraints.append({'type': 'ineq', 'fun': self.opt_fun.ineq_fun, 'jac': self.opt_fun.ineq_jac})

        self.history_cost = []
        res = minimize(
            fun=self.opt_fun.objective_fun, x0=z0, method='SLSQP',
            jac=self.opt_fun.objective_grad, bounds=bounds, constraints=constraints,
            callback=lambda zk: self.history_cost.append(self.opt_fun.objective_fun(zk)),
            options=dict(maxiter=maxiter, ftol=ftol, disp=disp),
        )

        U_opt, theta_opt, tau_opt = self.problem.split_decision(res.x)

        # Everything below is converted back into the user's units. The arithmetic
        # lives in ProblemScaling so that these several call sites cannot drift
        # apart on which way the conversion goes.
        sc = self.opt_fun.scaling
        J_opt = sc.objective_to_user(self.opt_fun.objective_fun(res.x))
        eq_resid   = sc.eq_to_user(self.opt_fun.eq_fun(res.x)) if self.opt_fun.has_eq else None
        ineq_resid = sc.ineq_to_user(self.opt_fun.ineq_fun(res.x)) if self.opt_fun.has_ineq else None
        t_opt, X_opt = self._numeric_rollout(U_opt, theta_opt, tau_opt)

        return dict(
            scipy_result=res, U_opt=U_opt, theta_opt=theta_opt, J_opt=J_opt,
            tau_opt=tau_opt,
            T_opt=(None if tau_opt is None else float(np.sum(tau_opt))),
            t_opt=t_opt, X_opt=X_opt, eq_resid=eq_resid, ineq_resid=ineq_resid,
            path_viol=self._path_violation(t_opt, X_opt, U_opt, theta_opt),
            history_cost=sc.objective_to_user(np.array(self.history_cost)),
            scaling=sc,
        )

    def _path_violation(self, t, X, U, theta):
        """
        max h(t) over the grid points, for each path inequality. This is the direct
        diagnostic for how well the transcription held: ineq_resid only tells you
        whether gamma - int L_eps is non-negative, whereas this reports how much
        the ORIGINAL constraint h(t) <= 0 is actually violated.
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
                    # The user's hfun only supports AD types; skip the diagnostic quietly.
                    return None
                worst = max(worst, float(np.max(hv)))
            out.append(worst)
        return np.asarray(out, dtype=float)

    def _numeric_rollout(self, U, theta, tau=None):
        """
        Replay the optimal trajectory with the problem's dyn/integrator, for plotting.

        Under the time-scaling transform each segment has its own duration, so the
        time grid is the cumulative sum of tau rather than a uniform linspace.
        """
        p = self.problem
        N, nx, nu = p.N, p.nx, p.nu
        durations = p.segment_durations(tau)

        # Use the problem's hook to build the numeric initial state from theta;
        # fall back to the fixed x0 when there is none.
        if hasattr(p, "numeric_initial_state") and callable(p.numeric_initial_state):
            x = np.asarray(p.numeric_initial_state(theta), dtype=float)
        else:
            x = np.asarray(p.x0, dtype=float)

        X = np.zeros((N+1, nx), dtype=float)
        X[0] = x
        t = np.concatenate(([0.0], np.cumsum(durations)))

        step = p.integrator
        f_num = _bind_dyn_numeric(p.dyn, theta)

        for k in range(N):
            if nu == 1:
                uk = np.array([U[k]], dtype=float)
            else:
                uk = np.asarray(U[nu*k:nu*(k+1)], dtype=float)
            # substeps are handled inside the integrator
            x = step(x, uk, durations[k], f_num)
            X[k+1] = x

        return t, X
