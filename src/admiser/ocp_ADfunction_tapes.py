# ocp_ADfunction_tapes.py
import inspect
import numpy as np
import cppad_py

from .ocp_smooth_utils import L_eps, smooth_abs

def _call_builder(builder, au, atheta, problem):
    if builder is None:
        return np.array([], dtype=object)
    n_params = len(inspect.signature(builder).parameters)
    if n_params == 3:
        return builder(au, atheta, problem)
    if n_params == 2:
        return builder(au, problem)
    raise TypeError("Builder must accept (au, atheta, problem) or (au, problem)")

def _bind_dyn_with_theta_ad(dyn, atheta):
    n_params = len(inspect.signature(dyn).parameters)
    if n_params == 2:
        return lambda x, u: dyn(x, u)
    return lambda x, u: dyn(x, u, atheta)

def build_ad_tapes(Utheta_template, problem):
    """
    Record all AD tapes in one place:
      - objective_ad : d_fun(az) -> [J]
      - eq_ad        : d_fun(az) -> G(z)   (may be None when there are none)
      - ineq_ad      : d_fun(az) -> C(z)   (may be None when there are none)
    """
    problem.validate()

    nuN = problem.N * problem.nu
    has_params = problem.has_params()

    # The objective and the constraints must share one quadrature scheme;
    # otherwise SLSQP sees an objective and constraints built on different
    # discretisations, and the resulting KKT point is meaningless.
    scheme = problem.resolve_quad_scheme()

    # ---------- objective ----------
    az_obj = cppad_py.independent(Utheta_template)
    au_obj = az_obj[:nuN]
    atheta_obj = az_obj[nuN:nuN + (problem.ntheta if has_params else 0)] if has_params else None
    aJ = _call_builder(problem.objective_builder, au_obj, atheta_obj, problem)
    objective_ad = cppad_py.d_fun(az_obj, np.asarray(aJ, dtype=object))

    # ---------- are there equalities / inequalities at all? ----------
    any_eq   = (len(problem.term_eq_specs) + len(problem.int_eq_specs) + len(problem.path_eq_specs)) > 0 \
               or (problem.constraint_builder is not None)
    any_ineq = (len(problem.term_ineq_specs) + len(problem.int_ineq_specs) + len(problem.path_ineq_specs)) > 0

    eq_ad   = None
    ineq_ad = None

    # ---------- equalities (legacy constraint_builder + canonical eq) ----------
    if any_eq:
        az  = cppad_py.independent(Utheta_template)
        au  = az[:nuN]
        ath = az[nuN:nuN + (problem.ntheta if has_params else 0)] if has_params else None

        # initial state (AD)
        if hasattr(problem, "ad_initial_state") and callable(problem.ad_initial_state):
            x = problem.ad_initial_state(ath)
        else:
            x = np.array([cppad_py.a_double(v) for v in problem.x0], dtype=object)

        t = cppad_py.a_double(0.0)

        f = _bind_dyn_with_theta_ad(problem.dyn, ath)

        # accumulators for the integral terms
        eq_int_acc    = [cppad_py.a_double(0.0) for _ in problem.int_eq_specs]
        path_eq_acc   = [cppad_py.a_double(0.0) for _ in problem.path_eq_specs]

        def acc_cb(t_sub, x_sub, u_sub, w_sub):
            w_ad = w_sub if isinstance(w_sub, cppad_py.a_double) else cppad_py.a_double(float(w_sub))
            _t   = t_sub if (t_sub is not None) else t
            # integral equalities
            for i, spec in enumerate(problem.int_eq_specs):
                q = spec["qfun"](_t, x_sub, u_sub, ath)
                eq_int_acc[i] = eq_int_acc[i] + w_ad * q
            # path equalities: int h^2 dt, or int smooth|h| dt
            for j, spec in enumerate(problem.path_eq_specs):
                h = spec["hfun"](_t, x_sub, u_sub, ath)
                if spec["mode"] == "L2":
                    integrand = h * h
                else:
                    integrand = smooth_abs(h, spec["eps_abs"])
                path_eq_acc[j] = path_eq_acc[j] + w_ad * integrand

        # rollout
        for k in range(problem.N):
            uk = (np.array([au[k]], dtype=object) if problem.nu == 1
                  else np.asarray(au[problem.nu*k : problem.nu*(k+1)], dtype=object))
            dt_k = problem.dt_of_segment(k)
            x = problem.integrator(
                x, uk, dt_k, f,
                accumulate_cb=acc_cb, t0=t,
                quad=scheme,
            )
            t = t + dt_k

        # terminal state
        xT = x

        # assemble the equalities: legacy + canonical
        G = []

        # legacy path, if constraint_builder is still supplied
        if problem.constraint_builder is not None:
            aC_legacy = _call_builder(problem.constraint_builder, au, ath, problem)
            aC_legacy = np.atleast_1d(np.asarray(aC_legacy, dtype=object))
            if aC_legacy.size > 0:
                G.extend(list(aC_legacy))

        # terminal equalities
        for spec in problem.term_eq_specs:
            psi = spec["psi"](xT, ath)
            psi = np.atleast_1d(np.asarray(psi, dtype=object))
            G.extend(list(psi))

        # integral equalities
        for i, spec in enumerate(problem.int_eq_specs):
            target = cppad_py.a_double(spec["target"])
            G.append(eq_int_acc[i] - target)

        # path equalities (the integral must vanish)
        for j, _ in enumerate(problem.path_eq_specs):
            G.append(path_eq_acc[j])

        G_arr = np.asarray(G, dtype=object)
        if G_arr.size > 0:
            eq_ad = cppad_py.d_fun(az, G_arr)

    # ---------- inequalities ----------
    if any_ineq:
        az  = cppad_py.independent(Utheta_template)
        au  = az[:nuN]
        ath = az[nuN:nuN + (problem.ntheta if has_params else 0)] if has_params else None

        # initial state (AD)
        if hasattr(problem, "ad_initial_state") and callable(problem.ad_initial_state):
            x = problem.ad_initial_state(ath)
        else:
            x = np.array([cppad_py.a_double(v) for v in problem.x0], dtype=object)

        t = cppad_py.a_double(0.0)

        f = _bind_dyn_with_theta_ad(problem.dyn, ath)

        ineq_int_acc = [cppad_py.a_double(0.0) for _ in problem.int_ineq_specs]
        path_ineq_acc= [cppad_py.a_double(0.0) for _ in problem.path_ineq_specs]

        def acc_cb(t_sub, x_sub, u_sub, w_sub):
            w_ad = w_sub if isinstance(w_sub, cppad_py.a_double) else cppad_py.a_double(float(w_sub))
            _t   = t_sub if (t_sub is not None) else t
            # integral inequalities
            for i, spec in enumerate(problem.int_ineq_specs):
                q = spec["qfun"](_t, x_sub, u_sub, ath)
                ineq_int_acc[i] = ineq_int_acc[i] + w_ad * q
            # Path-inequality transcription. eps is the value currently in
            # effect: a continuation solve scales it temporarily through
            # OCPProblem.scaled_eps(), leaving the registered value untouched.
            for j, spec in enumerate(problem.path_ineq_specs):
                h = spec["hfun"](_t, x_sub, u_sub, ath)
                path_ineq_acc[j] = path_ineq_acc[j] + w_ad * L_eps(h, problem.effective_eps(spec))

        # rollout
        for k in range(problem.N):
            uk = (np.array([au[k]], dtype=object) if problem.nu == 1
                  else np.asarray(au[problem.nu*k : problem.nu*(k+1)], dtype=object))
            dt_k = problem.dt_of_segment(k)
            x = problem.integrator(
                x, uk, dt_k, f,
                accumulate_cb=acc_cb, t0=t,
                quad=scheme,
            )
            t = t + dt_k

        xT = x

        # assemble the inequalities as C(z) >= 0
        C = []

        # terminal inequalities: phi <= 0 -> -phi >= 0;  phi >= 0 -> +phi >= 0
        for spec in problem.term_ineq_specs:
            phi   = spec["phi"](xT, ath)
            phi   = np.atleast_1d(np.asarray(phi, dtype=object))
            sense = spec["sense"]
            if sense == "<=":
                for v in phi: C.append(-v)
            else:
                for v in phi: C.append(v)

        # integral inequalities:  '<=' -> bound - int q >= 0;  '>=' -> int q - bound >= 0
        for i, spec in enumerate(problem.int_ineq_specs):
            b = cppad_py.a_double(spec["bound"])
            if spec["sense"] == "<=":
                C.append(b - ineq_int_acc[i])
            else:
                C.append(ineq_int_acc[i] - b)

        # path inequalities:  gamma - int L_eps >= 0
        for j, spec in enumerate(problem.path_ineq_specs):
            gamma = cppad_py.a_double(problem.effective_gamma(spec))
            C.append(gamma - path_ineq_acc[j])

        C_arr = np.asarray(C, dtype=object)
        if C_arr.size > 0:
            ineq_ad = cppad_py.d_fun(az, C_arr)

    return objective_ad, eq_ad, ineq_ad
