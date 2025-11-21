# optimiza_fun_class.py

import numpy as np

class optimize_fun_class:
    """
    把 cppad_py.d_fun 封装成 SciPy minimize + SLSQP 友好的接口。

    - objective_ad: au -> [J]
    - constraint_ad: au -> g(U) , shape (m,)

    暴露:
    - objective_fun(U) -> float
    - objective_grad(U) -> grad shape (n,)
    - constraint_fun(U) -> g(U) shape (m,)
    - constraint_jac(U) -> Jacobian dg/dU shape (m,n)
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
    # SciPy: type='ineq' 需要 fun(x) >= 0
    def ineq_fun(self, x):
        return self.ineq_ad.forward(0, x) if self.ineq_ad is not None else np.array([], dtype=float)

    def ineq_jac(self, x):
        return self.ineq_ad.jacobian(x) if self.ineq_ad is not None else np.zeros((0, x.size))

