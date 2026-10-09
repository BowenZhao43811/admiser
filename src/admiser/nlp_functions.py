# nlp_functions.py

import numpy as np

from .problem_scaling import identity_scaling


class NLPFunctions:
    """
    Present the transcribed NLP through the interface SciPy's minimize/SLSQP expects.

    The compiled function built by ocp_to_nlp returns all outputs stacked as
    [J ; G ; C], so one evaluation yields the objective AND every constraint, and
    one Jacobian yields every derivative. This class slices that one result into
    the six callbacks SciPy asks for.

    One round of the transcription
    ------------------------------
    eps and gamma -- one value per path inequality -- are fixed for the lifetime of
    this object and handed to the compiled function on every call. A continuation
    solve simply creates a new NLPFunctions per round; the compiled function, and
    its compilation, are shared by all of them.

    Scaling
    -------
    The compiled function always computes the problem in the USER's units. Any
    rescaling is applied here, on the way out to SciPy, and nowhere else; the
    solver converts the numbers it reports back with ProblemScaling.*_to_user.
    Keeping the compiled function unscaled is what lets to_nlp() hand back the
    user's own problem and lets the scaling change without recompiling anything.

    Caching
    -------
    SLSQP evaluates the objective, its gradient, the constraints and their
    Jacobians all at the SAME z before it moves on. Without caching each of those
    six calls would run the whole simulation again. So the last values and the
    last Jacobian are remembered, keyed on z: the first call at a new point does
    the work, the others read the stored result.

    Exposes
    -------
    objective_fun(z)  -> float
    objective_grad(z) -> gradient, shape (n,)
    eq_fun(z)         -> G(z), shape (n_eq,)
    eq_jac(z)         -> dG/dz, shape (n_eq, n)
    ineq_fun(z)       -> C(z), shape (n_ineq,)
    ineq_jac(z)       -> dC/dz, shape (n_ineq, n)
    """

    def __init__(self, compiled, eps, gamma, scaling=None):
        self.compiled = compiled

        # This round's transcription parameters, one value per path inequality.
        self.eps = np.asarray(eps, dtype=float)
        self.gamma = np.asarray(gamma, dtype=float)

        self.n_eq = compiled.n_eq
        self.n_ineq = compiled.n_ineq

        # None means "present the problem exactly as the user wrote it".
        self.scaling = scaling if scaling is not None else identity_scaling(
            compiled.n_eq, compiled.n_ineq)

        # Row ranges of the three blocks inside the stacked output vector.
        self._obj = compiled.obj_slice
        self._eq = compiled.eq_slice
        self._ineq = compiled.ineq_slice

        # Cache state: the point each cached result belongs to, and the results.
        self._z_values = None
        self._values = None
        self._z_jacobian = None
        self._jacobian = None

    # ---- convenience flags, so callers do not have to inspect the layout ----
    @property
    def has_eq(self) -> bool:
        return self.n_eq > 0

    @property
    def has_ineq(self) -> bool:
        return self.n_ineq > 0

    # ---- cached evaluation ----
    # The compiled functions return JAX arrays; np.array(...) copies them into
    # ordinary NumPy arrays, which is what SciPy works with.
    def _all_values(self, z):
        """[J ; G ; C] at z, reusing the previous evaluation when z has not moved."""
        z = np.asarray(z, dtype=float)
        if self._z_values is None or not np.array_equal(z, self._z_values):
            self._values = np.array(self.compiled.values(z, self.eps, self.gamma), dtype=float)
            self._z_values = z.copy()
        return self._values

    def _all_jacobian(self, z):
        """d[J ; G ; C]/dz at z, reusing the previous one when z has not moved."""
        z = np.asarray(z, dtype=float)
        if self._z_jacobian is None or not np.array_equal(z, self._z_jacobian):
            _, jacobian = self.compiled.values_and_jacobian(z, self.eps, self.gamma)
            self._jacobian = np.array(jacobian, dtype=float)
            self._z_jacobian = z.copy()
        return self._jacobian

    # ---- objective ----
    # A single positive factor: it cannot move the minimiser, it only changes what
    # SLSQP's absolute ftol test means.
    def objective_fun(self, z):
        return float(self._all_values(z)[self._obj][0]) * self.scaling.objective

    def objective_grad(self, z):
        return self._all_jacobian(z)[self._obj].flatten() * self.scaling.objective

    # ---- equality constraints (empty arrays when the problem has none) ----
    # Row scaling: g_i = 0 and s_i*g_i = 0 describe the same set for any s_i > 0,
    # so the feasible set is untouched; only the QP's conditioning changes.
    def eq_fun(self, z):
        if not self.has_eq:
            return np.array([], dtype=float)
        return self._all_values(z)[self._eq] * self.scaling.eq

    def eq_jac(self, z):
        if not self.has_eq:
            return np.zeros((0, np.asarray(z).size))
        # one factor per row, so it multiplies down the rows
        return self._all_jacobian(z)[self._eq] * self.scaling.eq[:, None]

    # ---- inequality constraints ----
    # SciPy convention: type='ineq' requires fun(z) >= 0
    def ineq_fun(self, z):
        if not self.has_ineq:
            return np.array([], dtype=float)
        return self._all_values(z)[self._ineq] * self.scaling.ineq

    def ineq_jac(self, z):
        if not self.has_ineq:
            return np.zeros((0, np.asarray(z).size))
        return self._all_jacobian(z)[self._ineq] * self.scaling.ineq[:, None]
