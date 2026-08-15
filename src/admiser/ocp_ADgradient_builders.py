# ocp_ADgradient_builders.py

import numpy as np

class optimize_fun_class:
    """
    Wrap cppad_py.d_fun tapes into an interface that SciPy's minimize/SLSQP can use.

    Inputs
    ------
    objective_ad : tape mapping z -> [J]
    eq_ad        : tape mapping z -> G(z), shape (m_eq,), optional
    ineq_ad      : tape mapping z -> C(z), shape (m_ineq,), optional

    Exposes
    -------
    objective_fun(z)  -> float
    objective_grad(z) -> gradient, shape (n,)
    eq_fun(z)         -> G(z), shape (m_eq,)
    eq_jac(z)         -> dG/dz, shape (m_eq, n)
    ineq_fun(z)       -> C(z), shape (m_ineq,)
    ineq_jac(z)       -> dC/dz, shape (m_ineq, n)
    """
    def __init__(self, objective_ad, eq_ad=None, ineq_ad=None):
        self.objective_ad = objective_ad
        self.eq_ad = eq_ad
        self.ineq_ad = ineq_ad

    # ---- objective ----
    def objective_fun(self, x):
        return float(self.objective_ad.forward(0, x)[0])

    def objective_grad(self, x):
        J = self.objective_ad.jacobian(x)
        return J.flatten()

    # ---- equality (optional) ----
    def eq_fun(self, x):
        return self.eq_ad.forward(0, x) if self.eq_ad is not None else np.array([], dtype=float)

    def eq_jac(self, x):
        return self.eq_ad.jacobian(x) if self.eq_ad is not None else np.zeros((0, x.size))

    # ---- inequality (optional) ----
    # SciPy convention: type='ineq' requires fun(x) >= 0
    def ineq_fun(self, x):
        return self.ineq_ad.forward(0, x) if self.ineq_ad is not None else np.array([], dtype=float)

    def ineq_jac(self, x):
        return self.ineq_ad.jacobian(x) if self.ineq_ad is not None else np.zeros((0, x.size))
