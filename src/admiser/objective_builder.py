# objective_builder.py
r"""
make_builders only builds the objective_builder, i.e. \int L dt + Phi.

All constraints -- terminal, integral and path, equalities and inequalities --
are registered through the OCPProblem.add_* API, and build_ad_tape assembles
the objective and all constraints onto one tape from a single rollout.

The return value is a SINGLE objective_builder function (an early version
returned a 2-tuple; that is gone).
"""

import numpy as np
import cppad_py
import inspect

from .problem_definition import DEFAULT_QUAD_SCHEME


def _bind_dyn_with_theta_ad(dyn, atheta):
    """Wrap dyn into f(x, u) = dyn(x, u, atheta?) as an AD-side closure."""
    n_params = len(inspect.signature(dyn).parameters)
    if n_params == 2:
        return lambda x, u: dyn(x, u)
    return lambda x, u: dyn(x, u, atheta)

def make_builders(
    *,
    dyn,
    L=None,               # L(t, x, u, theta); may be None
    Phi=None,             # Phi(xT, theta); may be None
    # Kept only so old calls still work; no longer used to build constraints:
    terminal_eq=None,
    integral_eqs=None,
    quad: str = DEFAULT_QUAD_SCHEME,
):
    """
    Returns
    -------
    objective_builder(au, atheta, problem) -> np.array([J], dtype=object)

    `quad` is this objective's default substep quadrature scheme; the available
    names and their orders are in quadrature.QUAD_SCHEMES. If the
    problem sets problem.quad_scheme explicitly, that wins -- the objective and
    every constraint must share one scheme, otherwise SLSQP sees them built on
    different discretisations and the resulting KKT point is meaningless.
    """

    def objective_builder(au, atheta, problem):
        N, nu = problem.N, problem.nu

        # initial state (a_double)
        if hasattr(problem, "ad_initial_state") and callable(problem.ad_initial_state):
            x = problem.ad_initial_state(atheta)
        else:
            x = np.array([cppad_py.a_double(v) for v in problem.x0], dtype=object)

        t = cppad_py.a_double(0.0)
        J = cppad_py.a_double(0.0)

        f = _bind_dyn_with_theta_ad(dyn, atheta)

        def acc_cb(t_sub, x_sub, u_sub, w_sub):
            nonlocal J  # must be declared before J is first used in this scope
            if L is None:
                return
            # w_sub is a quadrature weight and may be float or a_double; normalise it
            w_ad = w_sub if isinstance(w_sub, cppad_py.a_double) else cppad_py.a_double(float(w_sub))
            _t   = t_sub if (t_sub is not None) else t
            Lval = L(_t, x_sub, u_sub, atheta)
            J    = J + w_ad * Lval

        scheme = problem.resolve_quad_scheme()

        # rollout with accumulation
        for k in range(N):
            if nu == 1:
                uk = np.array([au[k]], dtype=object)
            else:
                uk = np.asarray(au[nu*k : nu*(k+1)], dtype=object)
            dt_k = problem.dt_of_segment(k)
            x = problem.integrator(
                x, uk, dt_k, f,
                accumulate_cb=acc_cb, t0=t,
                quad=scheme,
            )
            t = t + dt_k

        # Terminal cost. Phi is allowed to be a function that returns None in some
        # cases, which is how the shipped template is written.
        if Phi is not None:
            Phi_val = Phi(x, atheta)
            if Phi_val is not None:
                J = J + Phi_val

        return np.array([J], dtype=object)

    # Expose the pieces the objective is made of, as attributes on the returned
    # function. Two consumers rely on them:
    #
    #   quad      OCPProblem.resolve_quad_scheme reads it, so the objective and
    #             the constraint rows end up on the same quadrature scheme. (In an
    #             earlier version this argument was unconditionally overridden and
    #             had no effect at all.)
    #   L, Phi    build_ad_tape accumulates int L dt inside its single shared
    #             rollout instead of calling this closure, which is what lets one
    #             rollout serve the objective and every constraint.
    #   dyn       build_ad_tape checks it against problem.dyn: if the two differ,
    #             the objective and the constraints would be built on different
    #             dynamics, so it falls back to calling this closure instead.
    #
    # The closure itself remains fully functional, and is still used whenever the
    # fast path does not apply.
    objective_builder.quad = quad
    objective_builder.L = L
    objective_builder.Phi = Phi
    objective_builder.dyn = dyn
    return objective_builder
