# ADMISER — A Modern Control-Parametrization OCP Solver (Python + CppAD_py)

ADMISER is a **research-grade optimal control** toolkit that implements Teo’s **Control Parametrization** methodology with modern Python ergonomics:
- ✅ **Control Parametrization** (piecewise-constant controls; policy/feedback parametrization)
- ✅ **Constraint Transcription** (smooth path inequalities → canonical integral constraints)
- ✅ **Canonical constraints**: terminal eq/ineq, integral eq/ineq, path eq/ineq (smoothed)
- ✅ **Multi-substep RK4** integrator with substep quadrature callbacks
- ✅ **Automatic Differentiation** via `cppad_py` tapes (objective + constraints Jacobians)
- ✅ **SQP** method for solving nonlinear programming problem

> ⚠️ **Prereq**: `cppad_py` requires a local build on Linux/WSL. ADMISER provides a helper script to build and install it.

---

## Table of Contents
- [Install](#install)
- [Quickstart](#quickstart)
- [Core Ideas](#core-ideas)
- [Problem Definition Template](#problem-definition-template)
- [Registering Constraints](#registering-constraints)
- [Free Terminal Time Transforme](#free-terminal-time)
- [Cite / Acknowledge](#cite--acknowledge)

---

## Install

### create venv.
Create venv to Keep a clear global environment 
```sh
python3 -m venv ~/admiser_venv
source ~/admiser_venv/bin/activate
pip install --upgrade pip
```

### install `ADMISER`
From the repository root:
```sh
pip install "admiser @ git+https://github.com/BowenZhao43811/admiser.git@v0.1.0"
```
> ⚠️ **Python version requirement**: Python ≥ 3.10

> ⚠️**Dependencies** `numpy`, `scipy`, `matplotlib`, `cppad_py` (Auto installed).

## Quickstart

The jupyter notebook document `main_solver.ipynb` distributed together with the package records the solution results of many examples from different industries. Definitions of these example problems can be found in the `admiser.examples.ProblemName.py`

### find the accompanying example solution 
```sh
from admiser.examples import get_notebook_path

path = get_notebook_path()
print("Notebook path:", path)
```
Grab the file path then open it with your prefered editor (don't forget to select your venv where admiser was installed).

> ℹ️ **Figure Output**: States and control trajectories ploting is not integrated into the solver, instead, the flaxibility is given to the users. User can access raw optimization reaults of control, system parameters, objective, state, residual on the cononical equality and inequalities by 
`res["U_opt"]`
`res.get("theta_opt", None)`
`res["J_opt"]`
`res["X_opt"]`
`res.get("eq_res", None)`
`res.get("ineq_res", None)`.

> 😊 However, a chart generator template `solve_A_my_problem_template.py` can be access in `admiser.templates`


## Core Ideas

- **Control Parametrization**: treat controls as decision variables over segments; states propagate by integrating dynamics.

- **Constraint Transcription**: 
    - *Path inequality* $h(t) ≤ 0$ → smooth hinge $L_ε(h)$; enforce $∫ L_ε(h) dt ≤ γ$ (with $γ ≥ 0$; typically $γ=ε/4$).
    - *Path equality* $h(t)=0$ → ∫ h^2 dt = 0$.

- **AD with `cppad_py`**: build AD tapes (computational graph) for the objective $G_0(z)$ and the $i-th$ constraints $G_i(z)$ **once**; obtain function value, gradient/Jacobian for `SLSQP`.

## Problem Definition Template

Create a problem file (e.g., `my_problem.py`) using the following simple template pattern (😊 Full template with all supported constraints `my_A_problem_temeplate.py` can be found in `admiser.templates`):
```py
import numpy as np
from functools import partial
from admiser import OCPProblem
from admiser import rk4_substeps
from admiser import make_builders

# Grid
T, N = 1.0, 100
dt   = T / N
nx, nu = 2, 1
x0   = np.array([0.0, -1.0], float)

# Dynamics and stage cost
def dyn(x, u, theta=None):
    x1, x2 = x
    return np.array([x2, -x1 + u[0]], dtype=object)

def L(t, x, u, theta):
    return x[0]*x[0] + x[1]*x[1] + 0.25*(u[0]*u[0])

# No terminal cost, no terminal equality, no integral equalities
objective_builder, constraint_builder = make_builders(
    dyn=dyn, L=L, Phi=None, terminal_eq=None, integral_eqs=None, quad='rk4-mid'
)

# Control bounds
def control_bounds_builder(p): return [(-10.0, 10.0)] * (p.N * p.nu)

# Assemble
problem = OCPProblem(
    N=N, dt=dt, x0=x0, dyn=dyn,
    integrator=partial(rk4_substeps, m_sub=10),
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    constraint_builder=constraint_builder,
    control_bounds_builder=control_bounds_builder,
)

__all__ = ["problem", "N", "dt"]

```

Then solve from a small driver script or from Jupyter Notebook:

```py
from admiser import OCPSolver
from my_problem import problem

res = OCPSolver(problem).build().solve(maxiter=800, ftol=1e-9, disp=True)
print(res["J_opt"])
```

## Registering Constraints

All constraints are canonicalized and handled uniformly inside the tapes:
- Terminal equality: $ψ(x_T, θ)=0$
    ```py
    problem.add_terminal_eq(lambda xT, th: xT[0] - cppad_py.a_double(1.0))  # x1(T) = 1
    ```
- Terminal inequality: $φ(x_T, θ) ≤ 0$
    ```py
    problem.add_terminal_ineq(lambda xT, th: xT[0] - cppad_py.a_double(2.0), sense="<=")  # x1(T) ≤ 2
    ```
- Integral equality: $∫ q(t,x,u,θ) dt = b$
    ```py
    problem.add_integral_eq(lambda t,x,u,th: x[0], target=1.0)
    ```
- Integral inequality: $∫ q(t,x,u,θ) dt ≤ b$
    ```py
    problem.add_integral_ineq(lambda t,x,u,th: u[0]*u[0], bound=5.0, sense="<=")
    ```

- Path inequality: $h(t,x,u,θ) ≤ b$
    ```py
    def h_ineq(t, x, u, th): return x[1] - 1.0
    problem.add_path_ineq(h_ineq, eps=1e-3, gamma=0.25e-3)
    ```
- Path equality: e.g. $h(t,x,u,θ) = 0$
    ```py
    def h_eq(t, x, u, th): return x[0]^2+0.36*u[1]-0.25
    problem.add_path_eq(h_eq, mode="L2")
    ```
> ℹ️ **Note** Path constraints are transformed into integral form by applying the constraints transcription techniques.
## Free Terminal Time