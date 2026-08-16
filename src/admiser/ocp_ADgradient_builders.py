# ocp_ADgradient_builders.py

import numpy as np


class optimize_fun_class:
    """
    Present the taped NLP through the interface SciPy's minimize/SLSQP expects.

    The tape produced by build_ad_tape holds all outputs stacked as
    [J ; G ; C], so a single forward sweep yields the objective AND every
    constraint, and a single Jacobian yields every derivative. This class slices
    that one result into the six callbacks SciPy asks for.

    Caching
    -------
    SLSQP evaluates the objective, its gradient, the constraints and their
    Jacobians all at the SAME z before it moves on. Without caching each of those
    six calls would replay the tape from scratch. So the last forward sweep and
    the last Jacobian are remembered, keyed on z: the first call at a new point
    does the work, the other five read the stored result.

    Exposes
    -------
    objective_fun(z)  -> float
    objective_grad(z) -> gradient, shape (n,)
    eq_fun(z)         -> G(z), shape (n_eq,)
    eq_jac(z)         -> dG/dz, shape (n_eq, n)
    ineq_fun(z)       -> C(z), shape (n_ineq,)
    ineq_jac(z)       -> dC/dz, shape (n_ineq, n)
    """

    def __init__(self, taped):
        self.taped = taped
        self.fun = taped.fun

        self.n_eq = taped.n_eq
        self.n_ineq = taped.n_ineq

        # Row ranges of the three blocks inside the stacked output vector.
        self._obj = taped.obj_slice
        self._eq = taped.eq_slice
        self._ineq = taped.ineq_slice

        # Cache state: the point each cached result belongs to, and the results.
        self._z_values = None
        self._values = None
        self._z_jacobian = None
        self._jacobian = None

    # ---- convenience flags, so callers do not have to inspect the tape ----
    @property
    def has_eq(self) -> bool:
        return self.n_eq > 0

    @property
    def has_ineq(self) -> bool:
        return self.n_ineq > 0

    # ---- cached tape evaluation ----
    def _all_values(self, z):
        """[J ; G ; C] at z, reusing the previous sweep when z has not moved."""
        z = np.asarray(z, dtype=float)
        if self._z_values is None or not np.array_equal(z, self._z_values):
            self._values = self.fun.forward(0, z)
            self._z_values = z.copy()
        return self._values

    def _all_jacobian(self, z):
        """d[J ; G ; C]/dz at z, reusing the previous one when z has not moved."""
        z = np.asarray(z, dtype=float)
        if self._z_jacobian is None or not np.array_equal(z, self._z_jacobian):
            self._jacobian = self.fun.jacobian(z)
            self._z_jacobian = z.copy()
        return self._jacobian

    # ---- objective ----
    def objective_fun(self, z):
        return float(self._all_values(z)[self._obj][0])

    def objective_grad(self, z):
        return self._all_jacobian(z)[self._obj].flatten()

    # ---- equality constraints (empty arrays when the problem has none) ----
    def eq_fun(self, z):
        if not self.has_eq:
            return np.array([], dtype=float)
        return self._all_values(z)[self._eq]

    def eq_jac(self, z):
        if not self.has_eq:
            return np.zeros((0, np.asarray(z).size))
        return self._all_jacobian(z)[self._eq]

    # ---- inequality constraints ----
    # SciPy convention: type='ineq' requires fun(z) >= 0
    def ineq_fun(self, z):
        if not self.has_ineq:
            return np.array([], dtype=float)
        return self._all_values(z)[self._ineq]

    def ineq_jac(self, z):
        if not self.has_ineq:
            return np.zeros((0, np.asarray(z).size))
        return self._all_jacobian(z)[self._ineq]
