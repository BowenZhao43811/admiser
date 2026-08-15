# ocp_integrators_utils.py
"""
Substep RK4 integrator, and the family of schemes used to integrate quantities
along the resulting trajectory.

Background
----------
Control parametrization holds the control constant on each segment
[t_k, t_k + dt) and advances the state through that segment with several RK4
substeps. Besides the state, several integrals along the same trajectory are
needed:

    objective            int L(t,x,u,theta) dt
    integral constraints int q(t,x,u,theta) dt
    path transcription   int L_eps(h(t,x,u,theta)) dt

None of these feed back into the state, so they can be accumulated while the
state is being advanced. The integrator does this by calling back at a number of
sample points, accumulate_cb(t_i, x_i, u, w_i); the caller accumulates
w_i * integrand, and summing over all substeps yields the integral.

Two independent accuracies
--------------------------
The state is always advanced by classical RK4 and is therefore 4th order. The
accuracy of the integrals, however, is decided separately by the sampling scheme
and is NOT automatically the same: an earlier version of this package sampled the
Euler half-step midpoint, giving a 4th-order state but only 2nd-order integrals.

The order of a scheme is limited by two things at once:

  1. the order of the quadrature rule itself (rectangle 1, midpoint/trapezoid 2,
     Simpson 4, ...)
  2. the accuracy of the STATE ESTIMATE at each sample point

The second is easy to overlook and is usually the binding one. Simpson's rule,
for instance, is 4th order, but if its midpoint state is only the Euler half-step
x + h/2*k1 (error O(h^2)) the whole thing falls back to 2nd order -- paying for
two extra integrand evaluations and buying no accuracy at all (see the note on
'simpson' below).

Design constraint: no extra dynamics evaluations
------------------------------------------------
Every scheme below reuses only the k1..k4 already computed by RK4 and the
intermediate states formed from them; none of them evaluate f again. The only
cost is the number of evaluations of the integrand g (n_eval), which drives the
size of the AD tape. So a higher order costs a bigger tape, not a second ODE solve.

The scheme family (orders are measured, on x' = x, g = t^2*x,
int_0^1 t^2 e^t dt = e - 2)
--------------------------------------------------------------------------------
    name          order  n_eval  sample points
    'left'          1      1     (t,       x)                     left rectangle
    'right'         1      1     (t+h,     x_end)                 right rectangle
    'midpoint'      2      1     (t+h/2,   x + h/2*k1)            midpoint rule
    'trapezoid'     2      2     both endpoints, weights h/2, h/2 trapezoid rule
    'simpson'       3      3     (t,x), (t+h/2, x + h/4*(k1+k2)), (t+h, x_end)
    'rk4'           4      4     the four RK4 stage points, weights h/6,h/3,h/3,h/6

On the midpoint used by 'simpson': it averages the two half-step estimates,
1/2*[(x + h/2*k1) + (x + h/2*k2)] = x + h/4*(k1 + k2). Their errors against the
true midpoint are exactly -/+ h^2/8 * f'f, so the leading terms cancel and the
averaged point is O(h^3) accurate, which lifts Simpson to 3rd order overall. Using
either half-step estimate alone leaves the whole scheme at 2nd order (measured
1.99 / 1.98) -- a direct demonstration of limitation 2 above.

On 'rk4': it is equivalent to treating the integrand as an augmented state
y' = g(t,x,u) and integrating it with the very same RK4 tableau. Since RK4 is 4th
order for any smooth ODE system, int g dt automatically matches the accuracy of
the state. Note that the fourth sample point must be the stage-4 argument
x + h*k3, NOT the step-end state x_end; substituting x_end breaks RK4's order
conditions and drops the scheme to 3rd order in practice.

QUAD_SCHEMES is the single source of scheme names; there are no aliases. Each
scheme has exactly one spelling package-wide, so a typo or an outdated name fails
loudly instead of being silently accepted as a different accuracy.
"""

from typing import NamedTuple

import numpy as np


class QuadScheme(NamedTuple):
    """
    Metadata for one quadrature scheme. `order` is the measured global order of
    convergence; `n_eval` drives the size of the AD tape.
    """
    order: int      # global order of convergence
    n_eval: int     # integrand evaluations per substep
    summary: str    # one-line description


#: All canonical scheme names -> metadata. Use quad_order(name) to query the order.
QUAD_SCHEMES = {
    'left':      QuadScheme(1, 1, "left rectangle: substep left endpoint"),
    'right':     QuadScheme(1, 1, "right rectangle: substep right endpoint (step-end state)"),
    'midpoint':  QuadScheme(2, 1, "midpoint rule: Euler half-step midpoint x + h/2*k1"),
    'trapezoid': QuadScheme(2, 2, "trapezoid rule: both endpoints, weights h/2, h/2"),
    'simpson':   QuadScheme(3, 3, "Simpson's rule with midpoint x + h/4*(k1+k2), the average "
                                  "of the two half-step estimates"),
    'rk4':       QuadScheme(4, 4, "RK4 augmented-state quadrature: four stage points, "
                                  "weights h/6, h/3, h/3, h/6"),
}


def validate_quad_scheme(name: str) -> str:
    """Validate a quadrature scheme name and return it unchanged; unknown names raise."""
    if name not in QUAD_SCHEMES:
        raise ValueError(
            f"unknown quad scheme {name!r}; expected one of {tuple(QUAD_SCHEMES)}. "
            "(A misspelled scheme name would silently skip the integral terms, "
            "so this raises instead.)"
        )
    return name


def quad_order(name: str) -> int:
    """Global order of convergence of the given quadrature scheme."""
    return QUAD_SCHEMES[validate_quad_scheme(name)].order


def _shift(t, delta):
    """Offset a time value, keeping None when the caller does not track time."""
    return None if t is None else t + delta


def _quad_samples(scheme, t, h, x, k1, k2, k3, x_end):
    """
    Sample points for one substep [t, t+h], as a tuple of (t_i, x_i, w_i).
    Accumulating sum(w_i * g(t_i, x_i)) approximates int g dt over that substep.

    Invariant: sum(w_i) == h, since every scheme integrates over the same span.
    k1..k3 and x_end are supplied for free by the RK4 step, so f is never
    evaluated here.
    """
    t_mid = _shift(t, 0.5 * h)
    t_end = _shift(t, h)

    if scheme == 'left':
        return ((t, x, h),)

    if scheme == 'right':
        return ((t_end, x_end, h),)

    if scheme == 'midpoint':
        # Euler half-step midpoint: error O(h^2), giving 2nd order with the midpoint rule
        return ((t_mid, x + 0.5 * h * k1, h),)

    if scheme == 'trapezoid':
        return ((t,     x,     0.5 * h),
                (t_end, x_end, 0.5 * h))

    if scheme == 'simpson':
        # Average of the two half-step estimates; leading errors cancel, giving an
        # O(h^3) midpoint and 3rd order overall.
        x_mid = x + 0.25 * h * (k1 + k2)
        return ((t,     x,     h / 6.0),
                (t_mid, x_mid, 4.0 * h / 6.0),
                (t_end, x_end, h / 6.0))

    if scheme == 'rk4':
        # The four RK4 stage points. The last one is the stage-4 argument
        # x + h*k3, not the step-end state x_end.
        return ((t,     x,                 h / 6.0),
                (t_mid, x + 0.5 * h * k1,  h / 3.0),
                (t_mid, x + 0.5 * h * k2,  h / 3.0),
                (t_end, x + h * k3,        h / 6.0))

    raise ValueError(f"unhandled quad scheme {scheme!r}")   # pragma: no cover


def rk4_step(x, u, h, dyn):
    """
    A single classical fourth-order Runge-Kutta step, without quadrature
    callbacks, for standalone use:

        x_{k+1} = x_k + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    x, u : float / numpy.array / arrays holding cppad_py.a_double
    dyn  : the user's dynamics, dyn(x, u) -> xdot
    """
    k1 = dyn(x, u)
    k2 = dyn(x + 0.5 * h * k1, u)
    k3 = dyn(x + 0.5 * h * k2, u)
    k4 = dyn(x + h * k3,       u)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_substeps(x, u, dt, f, m_sub=5, accumulate_cb=None, t0=None, quad='rk4'):
    """
    Multi-substep RK4 integrator, optionally calling back within each substep
    according to the chosen quadrature scheme, so that integrals along the
    trajectory can be accumulated.

    Parameters
    ----------
    x : ndarray (nx,)
        State at the start of the segment; elements may be float or cppad_py.a_double.
    u : ndarray (nu,)
        The (piecewise-constant) control on this segment.
    dt : float | a_double
        Total duration of this segment.
    f : callable
        f(x, u) -> dx/dt
    m_sub : int
        Number of substeps; the segment is split evenly into m_sub steps of h = dt/m_sub.
    accumulate_cb : callable | None
        accumulate_cb(t_i, x_i, u, w_i); the caller accumulates w_i * integrand.
        When None, quadrature is skipped entirely (pure state propagation, e.g.
        the numeric rollout after solving).
        Note the fourth argument is a QUADRATURE WEIGHT, not the substep length:
        a substep may fire the callback several times, and the weights sum to h.
    t0 : float | a_double | None
        Start time of this segment. When None, the callback receives t_i = None,
        which is fine if the integrand does not depend on t.
    quad : str
        Quadrature scheme name, see QUAD_SCHEMES. Default 'rk4' (4th order,
        matching the state).

    Returns
    -------
    x_{k+1} : state at the end of the segment

    Notes
    -----
    The state is advanced by full RK4 regardless of the scheme, so the END STATE
    DOES NOT DEPEND ON `quad`; the scheme only affects the accuracy of the
    integrals. Every scheme reuses k1..k4 and never evaluates f again.
    """
    scheme = validate_quad_scheme(quad)

    h = dt / m_sub
    t = t0

    for _ in range(m_sub):
        k1 = f(x, u)
        k2 = f(x + 0.5 * h * k1, u)
        k3 = f(x + 0.5 * h * k2, u)
        k4 = f(x + h * k3,       u)

        x_end = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if accumulate_cb is not None:
            for t_i, x_i, w_i in _quad_samples(scheme, t, h, x, k1, k2, k3, x_end):
                accumulate_cb(t_i, x_i, u, w_i)

        x = x_end
        t = _shift(t, h)

    return x
