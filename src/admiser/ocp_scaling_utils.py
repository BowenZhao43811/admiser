# ocp_scaling_utils.py
"""
Automatic problem scaling.

Why this exists
---------------
SLSQP's `ftol` is an ABSOLUTE test on the change in the objective. A problem whose
objective is of order 1e6 therefore cannot use it: ftol=1e-9 asks for a relative
accuracy of 1e-15, below machine precision, so the line search reports "no descent
direction" after a handful of iterations and stops -- at a point that may not even
be feasible. That is not hypothetical; it is exactly what examples/my_medical1.py
used to do, halting after 9 iterations with its budget constraint violated by 1e-2.

Multiplying the objective by a positive constant does not move the minimiser, but
it does change what `ftol` means. Normalising the objective to order 1 turns
SLSQP's absolute test into an effectively RELATIVE one -- which is what users
assume `ftol` already is.

The constraints get the same treatment for a different reason. They are a vector
whose rows can differ wildly in magnitude (in my_medical1 the dose budget is of
order 20 while the transcribed path integral is of order 1e-8), and that
imbalance is carried straight into the QP subproblem SLSQP solves at every
iteration. Row scaling is the standard remedy.

Two different rules, one idea
-----------------------------
The objective and the constraints are NOT scaled by the same formula:

  objective    one global factor, from the VALUE, 1/|J|
               because `ftol` compares values

  constraints  one factor PER ROW, from the JACOBIAN, 1/||grad g_i||
               because the QP's conditioning depends on gradients, and because a
               constraint that happens to be satisfied at the starting point has
               g_i ~ 0 there -- normalising by its value would divide by zero.
               Constraints being zero at the start is normal, even desirable.

What is NOT scaled
------------------
The AD tape always holds the user's problem in the user's own units. Scaling is
applied only where the NLP is handed to SciPy, and undone on everything reported
back. So `to_nlp()` returns the unscaled problem, changing the scaling never
requires re-recording a tape, and every number in the result dictionary is in the
units the user wrote.

Determinism
-----------
The factors are estimated by probing a few points around the initial guess. Those
probes are FIXED offsets, never random: the same problem must always produce the
same factors, or two runs would silently optimise differently scaled problems and
their results would not be comparable.
"""

from typing import NamedTuple

import numpy as np

#: Magnitudes below this are treated as "cannot estimate", and the corresponding
#: factor falls back to 1. Guards against dividing by an objective or a Jacobian
#: row that happens to vanish at the probe points.
MAGNITUDE_FLOOR = 1e-12

#: Factors are clamped to this range. A factor outside it would mean the estimate
#: itself is untrustworthy, and an extreme rescaling does more harm than good.
FACTOR_LIMITS = (1e-8, 1e8)


class ProblemScaling(NamedTuple):
    """
    The factors applied when handing the NLP to SciPy, plus how they were chosen.

    objective : multiply J by this
    eq        : per-row factors for G(z), shape (n_eq,)
    ineq      : per-row factors for C(z), shape (n_ineq,)

    The *_to_user helpers below are the ONLY place the division happens, so the
    several places that report results cannot drift apart on which direction the
    conversion goes.
    """
    objective: float
    eq: np.ndarray
    ineq: np.ndarray
    objective_mode: str      # "auto" | "manual" | "none"
    constraint_mode: str     # "auto" | "none"

    # ---- converting solver-side numbers back into the user's units ----
    def objective_to_user(self, value):
        return value / self.objective

    def eq_to_user(self, residual):
        if residual is None:
            return None
        return np.asarray(residual, dtype=float) / self.eq

    def ineq_to_user(self, residual):
        if residual is None:
            return None
        return np.asarray(residual, dtype=float) / self.ineq

    @property
    def is_identity(self) -> bool:
        return (self.objective == 1.0
                and np.all(self.eq == 1.0)
                and np.all(self.ineq == 1.0))

    def describe(self) -> str:
        """One line stating what was done, for the solver to print."""
        if self.is_identity:
            return "[ADMISER] scaling: none (objective and constraints left as written)"
        bits = []
        if self.objective != 1.0:
            bits.append(f"objective x{self.objective:.3e} ({self.objective_mode})")
        else:
            bits.append(f"objective unscaled ({self.objective_mode})")
        n_scaled = int(np.sum(self.eq != 1.0) + np.sum(self.ineq != 1.0))
        n_rows = self.eq.size + self.ineq.size
        if n_rows:
            lo = float(min(self.eq.min(initial=1.0), self.ineq.min(initial=1.0)))
            hi = float(max(self.eq.max(initial=1.0), self.ineq.max(initial=1.0)))
            bits.append(f"constraints {n_scaled}/{n_rows} rows scaled, "
                        f"factors in [{lo:.2e}, {hi:.2e}] ({self.constraint_mode})")
        return ("[ADMISER] scaling applied: " + "; ".join(bits)
                + "\n[ADMISER]   all reported values are converted back to your units")


def identity_scaling(n_eq: int, n_ineq: int) -> ProblemScaling:
    """A scaling that does nothing; used when scaling is switched off."""
    return ProblemScaling(objective=1.0,
                          eq=np.ones(n_eq), ineq=np.ones(n_ineq),
                          objective_mode="none", constraint_mode="none")


def _clamp(factor: float) -> float:
    """Keep a factor inside FACTOR_LIMITS."""
    lo, hi = FACTOR_LIMITS
    return float(min(max(factor, lo), hi))


def _fallback_probes(z0, bounds):
    """
    A small, FIXED set of points just off z0, used ONLY when the magnitude at z0
    itself is unusable.

    Some starting guesses are symmetric or flat points where the objective and
    even its gradient vanish -- examples/my_port_kobe.py starts exactly there,
    with J(z0) = 0 and grad J(z0) = 0 -- and 1/|J(z0)| would divide by zero. A
    small nudge is enough to break that symmetry.

    The nudge is deliberately SMALL: one percent of the box width, or one percent
    of |z0| when a variable is unbounded. An earlier version stepped a quarter of
    the way across the box and took the largest magnitude it saw, which was a bad
    mistake -- for a stiff closed loop such a probe lands in a blow-up region,
    reports an enormous |J|, and the resulting factor of ~1e-8 made ftol so loose
    that the solver stopped at the initial guess. The magnitude that matters is
    the one at the point the optimiser actually starts from.

    The offsets are deterministic, never random: the same problem must always
    produce the same factors, or two runs would optimise differently scaled
    problems and their results would not be comparable.
    """
    z0 = np.asarray(z0, dtype=float)
    lo = np.array([b[0] if np.isfinite(b[0]) else -np.inf for b in bounds], dtype=float)
    hi = np.array([b[1] if np.isfinite(b[1]) else np.inf for b in bounds], dtype=float)

    step = np.where(np.isfinite(lo) & np.isfinite(hi),
                    0.01 * (hi - lo),
                    0.01 * np.maximum(1.0, np.abs(z0)))

    return [np.clip(z0 + step, lo, hi),
            np.clip(z0 - step, lo, hi)]


def compute_scaling(unscaled_nlp, z0, bounds, objective="auto", constraints="auto") -> ProblemScaling:
    """
    Work out the scaling factors for a problem.

    Parameters
    ----------
    unscaled_nlp : an optimize_fun_class built WITHOUT scaling, i.e. presenting the
                   problem in the user's own units. Everything measured here is
                   therefore measured on what the user actually wrote.
    z0           : the initial guess, the point the probes are built around
    bounds       : the box bounds, used to size the probe offsets
    objective    : "auto", "none", or an explicit positive float
    constraints  : "auto" or "none"
    """
    n_eq, n_ineq = unscaled_nlp.n_eq, unscaled_nlp.n_ineq
    z0 = np.asarray(z0, dtype=float)
    fallbacks = _fallback_probes(z0, bounds)

    # ---------------- objective: one factor, from the value ----------------
    if isinstance(objective, (int, float)) and not isinstance(objective, bool):
        if objective <= 0.0:
            raise ValueError(f"an explicit objective scale must be positive, got {objective}")
        obj_factor, obj_mode = float(objective), "manual"
    elif objective == "none":
        obj_factor, obj_mode = 1.0, "none"
    elif objective == "auto":
        # Normalise |J| at the initial guess: that is the point the optimiser
        # starts from, so it is the magnitude ftol will be compared against.
        magnitude = abs(unscaled_nlp.objective_fun(z0))
        if magnitude < MAGNITUDE_FLOOR:
            # Degenerate start (J vanishes there). Nudge slightly and try again.
            magnitude = max(abs(unscaled_nlp.objective_fun(z)) for z in fallbacks)
        if magnitude < MAGNITUDE_FLOOR:
            # Still nothing to normalise against; leave the objective alone rather
            # than invent an enormous factor.
            obj_factor, obj_mode = 1.0, "auto"
        else:
            obj_factor, obj_mode = _clamp(1.0 / magnitude), "auto"
    else:
        raise ValueError(
            f"objective scaling must be 'auto', 'none' or a positive number, got {objective!r}")

    # ---------------- constraints: one factor per row, from the Jacobian ----------------
    if constraints == "none":
        eq_factors = np.ones(n_eq)
        ineq_factors = np.ones(n_ineq)
        con_mode = "none"
    elif constraints == "auto":
        con_mode = "auto"

        def row_factors(jac_of, n_rows):
            if n_rows == 0:
                return np.ones(0)
            # Largest absolute entry in each row: the infinity norm is the classic
            # choice for row preconditioning, being cheap and independent of the
            # number of variables. Measured at z0, for the same reason as the
            # objective; rows that are flat there fall back to a small nudge.
            def norms_at(z):
                J = np.atleast_2d(np.asarray(jac_of(z), dtype=float))
                return np.max(np.abs(J), axis=1)

            norms = norms_at(z0)
            flat = norms < MAGNITUDE_FLOOR
            if np.any(flat):
                for z in fallbacks:
                    norms[flat] = np.maximum(norms[flat], norms_at(z)[flat])

            out = np.ones(n_rows)
            usable = norms >= MAGNITUDE_FLOOR
            out[usable] = [_clamp(1.0 / n) for n in norms[usable]]
            return out

        eq_factors = row_factors(unscaled_nlp.eq_jac, n_eq)
        ineq_factors = row_factors(unscaled_nlp.ineq_jac, n_ineq)
    else:
        raise ValueError(
            f"constraint scaling must be 'auto' or 'none', got {constraints!r}")

    return ProblemScaling(objective=obj_factor,
                          eq=eq_factors, ineq=ineq_factors,
                          objective_mode=obj_mode, constraint_mode=con_mode)
