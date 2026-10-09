# sqp_solver.py

import numpy as np
from scipy.optimize import minimize

from .ocp_to_nlp import build_nlp
from .nlp_functions import NLPFunctions
from .problem_scaling import compute_scaling


class OCPSolver:
    """
    General optimal control solver.

    ------------- only two public methods -------------
    solve(...)   Solve. Whether that is a single solve or an eps -> 0 continuation
                 is decided by the problem itself, declared with
                 OCPProblem.set_transcription(); the solving side never branches on it.
    to_nlp(...)  Only transcribe the OCP into an NLP and return an object that
                 evaluates it and its derivatives -- NO optimisation. Useful for
                 checking gradients, e.g. against finite differences.

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
    - solve() never mutates the problem: each round's eps and gamma are handed to
      the compiled NLP as arguments, so the same problem can be solved repeatedly
      with identical results.
    """

    def __init__(self, problem: object):
        self.problem = problem
        self.compiled = None   # the CompiledNLP built by _compile()
        self.opt_fun = None    # SciPy-facing view of it, for the current round
        # Scaling factors, computed once in the first round and then FROZEN. They
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
        schedule = p.transcription_rounds()      # one (eps, gamma) pair per round
        if verbose is None:
            verbose = len(schedule) > 1

        z = p.initial_guess() if z0 is None else np.asarray(z0, dtype=float)
        rounds, res = [], None

        # Compiled once; every round below reuses it, because a round differs
        # from the previous one only in the VALUES of eps and gamma.
        self._compile()

        for k, (eps, gamma) in enumerate(schedule):
            self.opt_fun = self._nlp_view(eps, gamma, scaled=True)
            # Say once, up front, that the numbers were rescaled and by how much.
            # An automatic transform that changes the iteration history must never
            # be invisible.
            if verbose and k == 0:
                print(self.scaling.describe())
            res = self._solve_once(z, maxiter=maxiter, ftol=ftol, disp=disp)
            z = res["scipy_result"].x
            eps_now, gam_now = eps.tolist(), gamma.tolist()

            viol = res["path_viol"]
            max_viol = float(np.max(viol)) if viol is not None and viol.size else float("nan")
            rounds.append(dict(eps=eps_now, gamma=gam_now, J_opt=res["J_opt"],
                               max_path_viol=max_viol,
                               status=res["scipy_result"].status,
                               nit=res["scipy_result"].nit))
            if verbose:
                e = f"{min(eps_now):.3e}" if eps_now else "-"
                v = "n/a" if not np.isfinite(max_viol) else f"{max_viol:+.3e}"
                print(f"[ADMISER] round {k+1}/{len(schedule)}  eps={e}  "
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

        eps : which eps to use. None (default) uses the eps the FIRST continuation
              round would actually use (in "single" mode that is the registered
              value itself). Passing a number applies it to every path constraint.
        """
        p = self.problem
        self._compile()
        eps_arr, gamma_arr = p.eps_gamma(factor=p.eps_factors()[0], eps=eps)
        # Deliberately unscaled: this is meant to be the user's own problem, so a
        # gradient checked against finite differences here is a gradient of what
        # the user wrote, not of an internally rescaled version.
        self.opt_fun = self._nlp_view(eps_arr, gamma_arr, scaled=False)
        return self.opt_fun

    # ================= internals =================

    def _compile(self):
        """
        Transcribe the problem into its NLP functions (see ocp_to_nlp). JAX
        compiles them on their first call; every later call -- every SLSQP
        iteration, every continuation round -- reuses that compilation.
        """
        self.compiled = build_nlp(self.problem)
        return self.compiled

    def _nlp_view(self, eps, gamma, scaled=True):
        """
        The SciPy-facing view of the compiled NLP for one round's eps and gamma.

        The compiled functions always work in the user's units. When `scaled` is
        set, the view multiplies by the scaling factors on the way out; those are
        estimated once, from an UNSCALED view of the first round, so they describe
        the problem as the user wrote it, and then stay frozen.
        """
        if not scaled:
            return NLPFunctions(self.compiled, eps, gamma)

        if self.scaling is None:
            cfg = getattr(self.problem, "scaling", None) or {}
            unscaled = NLPFunctions(self.compiled, eps, gamma)
            self.scaling = compute_scaling(
                unscaled, self.problem.initial_guess(), self.problem.make_bounds(),
                objective=cfg.get("objective", "auto"),
                constraints=cfg.get("constraints", "auto"),
            )
        return NLPFunctions(self.compiled, eps, gamma, self.scaling)

    def _solve_once(self, z0, maxiter, ftol, disp):
        """Solve the NLP once, with the current round's view of it."""
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
        # The reported trajectory comes from the very rollout that was optimised,
        # replayed at the solution -- there is no second simulation that could
        # disagree with it.
        t_opt, X_opt = (np.array(a, dtype=float) for a in self.compiled.trajectory(res.x))

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
                hv = np.atleast_1d(np.asarray(spec["hfun"](t[k], X[k], uk, theta), dtype=float))
                worst = max(worst, float(np.max(hv)))
            out.append(worst)
        return np.asarray(out, dtype=float)
