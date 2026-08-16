# ocp_ADfunction_tapes.py
"""
Record the optimal control problem onto a single AD tape.

What gets taped
---------------
One tape mapping the decision vector z to a stacked output vector

    [ J(z) ; G(z) ; C(z) ]

with J the objective, G the equality constraints (which must be zero) and C the
inequality constraints (which must be non-negative). The three blocks are read
back through the slices carried by the returned TapedNLP.

Why one tape and not three
--------------------------
All of these are integrals along the SAME state trajectory, so they can all be
accumulated during a single rollout. An earlier version of this module recorded
three separate tapes, each replaying the whole rollout. That cost about three
times the taping work, but the real problem was that the rollout logic then
existed in near-identical copies: a change applied to only one of them would give
the objective and the constraints different discretisations, which does not
raise -- it just produces a meaningless KKT point.

Where the objective integrand comes from
----------------------------------------
Builders produced by make_builders() carry their L, Phi and dyn as attributes, so
this module can accumulate int L dt inside the shared rollout. A hand-written
objective_builder that does not expose them still works: it is invoked while the
tape is recording, so its own rollout lands on this same tape. That route gives
one tape but two rollouts -- correct, just not the fast one.
"""

import inspect
from typing import NamedTuple

import numpy as np
import cppad_py

from .ocp_smooth_utils import L_eps, smooth_abs


class TapedNLP(NamedTuple):
    """
    The transcribed NLP: one tape plus the layout of its outputs.

    fun    : cppad_py.d_fun mapping z -> [J ; G ; C]
    n_eq   : number of equality rows
    n_ineq : number of inequality rows
    """
    fun: object
    n_eq: int
    n_ineq: int

    @property
    def obj_slice(self):
        """Rows holding the objective (always exactly one row)."""
        return slice(0, 1)

    @property
    def eq_slice(self):
        """Rows holding G(z)."""
        return slice(1, 1 + self.n_eq)

    @property
    def ineq_slice(self):
        """Rows holding C(z)."""
        return slice(1 + self.n_eq, 1 + self.n_eq + self.n_ineq)


def _call_builder(builder, au, atheta, problem):
    """Invoke a user-supplied builder, accepting either supported signature."""
    if builder is None:
        return np.array([], dtype=object)
    n_params = len(inspect.signature(builder).parameters)
    if n_params == 3:
        return builder(au, atheta, problem)
    if n_params == 2:
        return builder(au, problem)
    raise TypeError("Builder must accept (au, atheta, problem) or (au, problem)")


def _bind_dyn_with_theta_ad(dyn, atheta):
    """Wrap dyn into f(x, u), accepting either a 2- or 3-argument signature."""
    n_params = len(inspect.signature(dyn).parameters)
    if n_params == 2:
        return lambda x, u: dyn(x, u)
    return lambda x, u: dyn(x, u, atheta)


def _as_ad(value):
    """Normalise a float or an a_double into an a_double."""
    return value if isinstance(value, cppad_py.a_double) else cppad_py.a_double(float(value))


def build_ad_tape(z_template, problem) -> TapedNLP:
    """
    Record `problem` onto one tape and return it with its output layout.

    z_template supplies the SIZE of the decision vector and the point at which the
    tape is recorded; the finished tape can then be evaluated at any z.
    """
    problem.validate()

    nuN = problem.N * problem.nu
    has_params = problem.has_params()

    # The objective and every constraint must share one quadrature scheme.
    # Resolving it once, here, is what guarantees that.
    scheme = problem.resolve_quad_scheme()

    # ------------------------------------------------------------------
    # Can the objective be folded into the shared rollout?
    #
    # make_builders() attaches L / Phi / dyn to the builder it returns, which is
    # what makes the fast path possible. Two situations fall back to letting the
    # builder run its own rollout on this same tape:
    #   * a hand-written builder that never exposed L
    #   * a builder whose dyn is not the problem's dyn. That is unusual, but the
    #     previous implementation genuinely did give the objective and the
    #     constraints different dynamics in that case, and silently changing
    #     that behaviour would be worse than taking the slower route.
    # ------------------------------------------------------------------
    builder = problem.objective_builder
    L = getattr(builder, "L", None)
    Phi = getattr(builder, "Phi", None)
    fold_objective_in = (hasattr(builder, "L")
                         and getattr(builder, "dyn", None) is problem.dyn)

    # ---------------- start recording ----------------
    az = cppad_py.independent(z_template)
    au = az[:nuN]
    ath = az[nuN:nuN + (problem.ntheta if has_params else 0)] if has_params else None

    # ---------------- one accumulator per integral being taken ----------------
    # Each is updated inside acc_cb and read out after the rollout finishes.
    J = cppad_py.a_double(0.0)                                                # int L dt
    eq_int_acc    = [cppad_py.a_double(0.0) for _ in problem.int_eq_specs]    # int q dt
    path_eq_acc   = [cppad_py.a_double(0.0) for _ in problem.path_eq_specs]   # int h^2 dt
    ineq_int_acc  = [cppad_py.a_double(0.0) for _ in problem.int_ineq_specs]  # int q dt
    path_ineq_acc = [cppad_py.a_double(0.0) for _ in problem.path_ineq_specs] # int L_eps(h) dt

    t = cppad_py.a_double(0.0)   # running time; acc_cb reads it through the closure

    def acc_cb(t_sub, x_sub, u_sub, w_sub):
        """
        Called by the integrator at every quadrature sample point.

        w_sub is the quadrature WEIGHT of this sample, not the substep length: one
        substep may fire this callback several times, and the weights sum to the
        substep length. Accumulating w * integrand over every call therefore
        produces the integral.

        This single callback feeds every integral in the problem, which is exactly
        what allows one rollout to serve the objective and all the constraints.
        """
        nonlocal J
        w = _as_ad(w_sub)
        t_here = t_sub if (t_sub is not None) else t

        # objective:  int L dt
        if L is not None:
            J = J + w * L(t_here, x_sub, u_sub, ath)

        # integral equalities:  int q dt
        for i, spec in enumerate(problem.int_eq_specs):
            eq_int_acc[i] = eq_int_acc[i] + w * spec["qfun"](t_here, x_sub, u_sub, ath)

        # path equalities:  int h^2 dt, or int smooth|h| dt
        for j, spec in enumerate(problem.path_eq_specs):
            h = spec["hfun"](t_here, x_sub, u_sub, ath)
            integrand = h * h if spec["mode"] == "L2" else smooth_abs(h, spec["eps_abs"])
            path_eq_acc[j] = path_eq_acc[j] + w * integrand

        # integral inequalities:  int q dt
        for i, spec in enumerate(problem.int_ineq_specs):
            ineq_int_acc[i] = ineq_int_acc[i] + w * spec["qfun"](t_here, x_sub, u_sub, ath)

        # path inequalities, transcribed:  int L_eps(h) dt.
        # eps is the value in effect right now: a continuation solve scales it
        # temporarily via OCPProblem.scaled_eps(), leaving the registered value alone.
        for j, spec in enumerate(problem.path_ineq_specs):
            h = spec["hfun"](t_here, x_sub, u_sub, ath)
            path_ineq_acc[j] = path_ineq_acc[j] + w * L_eps(h, problem.effective_eps(spec))

    # ---------------- the single rollout ----------------
    x = problem.ad_initial_state(ath)
    f = _bind_dyn_with_theta_ad(problem.dyn, ath)

    # Attach acc_cb only when something actually needs integrating; without it the
    # integrator skips all the sampling work.
    needs_quadrature = (L is not None or problem.int_eq_specs or problem.path_eq_specs
                        or problem.int_ineq_specs or problem.path_ineq_specs)

    for k in range(problem.N):
        uk = (np.array([au[k]], dtype=object) if problem.nu == 1
              else np.asarray(au[problem.nu*k: problem.nu*(k+1)], dtype=object))
        dt_k = problem.dt_of_segment(k)
        x = problem.integrator(
            x, uk, dt_k, f,
            accumulate_cb=acc_cb if needs_quadrature else None,
            t0=t,
            quad=scheme,
        )
        t = t + dt_k

    xT = x   # terminal state, used by the terminal constraints below

    # ---------------- objective ----------------
    if fold_objective_in:
        # J already holds int L dt from the shared rollout; add the terminal cost.
        # Phi may return None, which is how the shipped template spells
        # "no terminal cost".
        if Phi is not None:
            Phi_val = Phi(xT, ath)
            if Phi_val is not None:
                J = J + Phi_val
        objective_rows = [J]
    else:
        # Compatibility route: run the builder while the tape is still recording,
        # so its own rollout is captured on this same tape.
        aJ = np.atleast_1d(np.asarray(_call_builder(builder, au, ath, problem), dtype=object))
        objective_rows = list(aJ)

    # ---------------- equalities G(z) = 0 ----------------
    # The order below is the order the residual vector is reported in, so it is
    # part of the observable behaviour and must not be shuffled.
    G = []

    # legacy hook, if a constraint_builder is still supplied
    if problem.constraint_builder is not None:
        aC_legacy = np.atleast_1d(
            np.asarray(_call_builder(problem.constraint_builder, au, ath, problem), dtype=object))
        if aC_legacy.size > 0:
            G.extend(list(aC_legacy))

    # terminal equalities  psi(xT, theta) = 0
    for spec in problem.term_eq_specs:
        psi = np.atleast_1d(np.asarray(spec["psi"](xT, ath), dtype=object))
        G.extend(list(psi))

    # integral equalities  int q dt = target
    for i, spec in enumerate(problem.int_eq_specs):
        G.append(eq_int_acc[i] - cppad_py.a_double(spec["target"]))

    # path equalities, whose integral must vanish
    for j in range(len(problem.path_eq_specs)):
        G.append(path_eq_acc[j])

    # ---------------- inequalities C(z) >= 0 ----------------
    C = []

    # terminal inequalities:  phi <= 0 becomes -phi >= 0;  phi >= 0 stays as is
    for spec in problem.term_ineq_specs:
        phi = np.atleast_1d(np.asarray(spec["phi"](xT, ath), dtype=object))
        if spec["sense"] == "<=":
            C.extend([-v for v in phi])
        else:
            C.extend(list(phi))

    # integral inequalities:  "<=" -> bound - int q >= 0;  ">=" -> int q - bound >= 0
    for i, spec in enumerate(problem.int_ineq_specs):
        b = cppad_py.a_double(spec["bound"])
        C.append(b - ineq_int_acc[i] if spec["sense"] == "<=" else ineq_int_acc[i] - b)

    # path inequalities:  gamma - int L_eps >= 0
    for j, spec in enumerate(problem.path_ineq_specs):
        C.append(cppad_py.a_double(problem.effective_gamma(spec)) - path_ineq_acc[j])

    # ---------------- stop recording ----------------
    outputs = np.asarray(objective_rows + G + C, dtype=object)
    fun = cppad_py.d_fun(az, outputs)

    # Let CppAD prune the recording: it removes operations whose results are never
    # used and merges duplicated subexpressions, of which a rollout produces many.
    #
    # This matters more here than it would with separate tapes. Reverse mode costs
    # one sweep of the WHOLE tape per output row, so with everything stacked on one
    # tape each row now traverses the accumulation code of every other row too.
    # Optimising the tape buys that back and then some.
    #
    # Measured: 25-47% fewer operations, and the full Jacobian 1.3-1.8x faster.
    # The call itself costs 12-26% of the taping time and pays for itself after
    # roughly 7-36 Jacobian evaluations, while a solve needs one per iteration.
    #
    # Caveat: optimising reassociates floating-point arithmetic, so results can
    # move in the last bits (measured 1e-18 to 1e-15 relative). That is invisible
    # for a well-conditioned problem but can shift where a struggling optimiser
    # stops.
    fun.optimize()

    return TapedNLP(fun=fun, n_eq=len(G), n_ineq=len(C))
