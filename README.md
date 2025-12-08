# ADMISER — Automatic Differentiation Ehanced Modern Control-Parametrization OCP Solver

ADMISER is a **Numerical Optimal Control** toolkit that incorporates **Automatic Differentiation** for gradient computation within [Professor Kok Lay Teo](https://sunwayuniversity.edu.my/school-of-mathematical-sciences/staff-profiles/professor-teo-kok-lay)'s **Control Parametrization** framework in Python.


- ✅ **Control Parametrization** (piecewise-constant controls; policy/feedback parametrization)
- ✅ **Constraint Transcription** (smooth path inequalities → canonical integral constraints)
- ✅ **Canonical constraints**: terminal eq/ineq, integral eq/ineq, path eq/ineq (smoothed)
- ✅ **Multi-substep RK4** integrator with substep quadrature callbacks
- ✅ **System Parameters** are simultaneously optimized within the same framework (if have)
- ✅ **Automatic Differentiation** via `cppad_py` tapes (objective + constraints Jacobians)
- ✅ **SQP** method for solving nonlinear programming problem

> ⚠️ **Strict prerequest**: `cppad_py` requires a local build on Linux/WSL. ADMISER provides a helper script to build and install it.

---

## Table of Contents
- [Install](#install)
- [Quickstart](#quickstart)
- [Core Ideas](#core-ideas)
- [Problem Template](#problem-template)
- [Registering Constraints](#registering-constraints)
- [Free Terminal Time Transforme](#free-terminal-time)
- [Cite / Acknowledge](#cite--acknowledge)

---

## Install

### install WSL on Windows
Install wsl and linux distributation.
```powershell
# install system components for WSL
wsl --install
# install Linux distributation on WSL
wsl.exe --install Ubuntu
```
> ⚠️ Minimum requirements for running the package are provided, for details on WSL please refer to Microsoft's official documentations.

Creat user in Linux system
```powershell
Create a default Unix user account: ****** (Input your user name)
# password won't show in Linux.
New password: ****** (Input your user password)
Retype new password: ****** (Reinput your user password)
passwd: password updated successfully # success!
```

### update Ubuntu system components
```sh
# Switch from windows file system to linux
cd ~
# update
sudo apt update
sudo apt upgrade -y
```

### install Python Develope tools
Install python, package management tools, vertial environment management tools, and python development tools on Linux `python3`, `pip`, `venv`, `dev`.
```sh
sudo apt install -y python3 python3-pip python3-venv python3-dev
```

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
pip install "admiser @ git+https://github.com/BowenZhao43811/admiser.git"
```
> ⚠️ **Python version requirement**: Python ≥ 3.10

> ⚠️**Dependencies** `numpy`, `scipy`, `matplotlib`, `cppad_py` (Auto installations are provided in the next subsection).

### install cppad and build cppad_py
```sh
admiser-install-cppad
```
> ⚠️ Minimum requirements for running the package are provided, for details on CppAD and CppAD_Py please refer to COIN's official documentations.

## Quickstart

The jupyter notebook document `main_solver.ipynb` distributed together with the package records the solution results of many examples from different industries. Definitions of these example problems can be found in the `admiser.examples.ProblemName.py`

### find the accompanying example solution 
```sh
from admiser.examples import get_notebook_path

path = get_notebook_path()
print("Notebook path:", path)
```
Grab the file path then open it with your prefered editor (don't forget to select your venv where admiser was installed).

## Core Ideas

- **Control Parametrization**: treat controls as decision variables over segments; states propagate by integrating dynamics.

- **Constraint Transcription**: 
    - *Path inequality* $h(t) ≤ 0$ → smooth hinge $L_ε(h)$; enforce $∫ L_ε(h) dt ≤ γ$ (with $γ ≥ 0$; typically $γ=ε/4$).
    - *Path equality* $h(t)=0$ → $∫ h^2 dt = 0$.

- **AD with `cppad_py`**: build AD tapes (computational graph) for the objective $G_0(z)$ and the $i-th$ constraints $G_i(z)$ **once**; obtain function value, gradient/Jacobian for `SLSQP`.

## Problem Template

### Problem define

Create a problem file (e.g., `my_problem.py`) using the following simple template pattern (😊 Full template with all supported constraints `my_A_problem_temeplate.py` can be found in `admiser\templates`):
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
u0 = 1

# Dynamics and stage cost
def dyn(x, u, theta=None):
    x1, x2 = x
    return np.array([x2, -x1 + u[0]], dtype=object)

def L(t, x, u, theta):
    return x[0]*x[0] + x[1]*x[1] + 0.25*(u[0]*u[0])

# No terminal cost, no terminal equality, no integral equalities
objective_builder = make_builders(dyn=dyn, L=L, Phi=None, terminal_eq=None, integral_eqs=None, quad='rk4-mid')

# Control bounds
def control_bounds_builder(p): return [(-10.0, 10.0)] * (p.N * p.nu)

# Assemble
problem = OCPProblem(
    N=N, dt=dt, x0=x0, u0=u0, dyn=dyn,
    integrator=partial(rk4_substeps, m_sub=10),
    nu=nu, nx=nx,
    objective_builder=objective_builder,
    control_bounds_builder=control_bounds_builder,
    ntheta=0)
```
### Problem solve

Then solve from a small driver script.

```py
from admiser import OCPSolver

res = OCPSolver(problem).build().solve(maxiter=800, ftol=1e-9, disp=True)
print(res["J_opt"])
```

> ℹ️ **Figure Output**: States and control trajectories ploting is not integrated into the solver, instead, the flaxibility is given to the users. User can access raw optimization reaults of control, system parameters, objective, state, residual on the cononical equality and inequalities by 
`res["U_opt"]`
`res.get("theta_opt", None)`
`res["J_opt"]`
`res["X_opt"]`
`res.get("eq_res", None)`
`res.get("ineq_res", None)`.

> 😊 However, a chart generator template `solve_A_my_problem_template.py` can be access in `admiser\templates`


## Registering Constraints

All constraints are canonicalized and handled uniformly inside `admiser` as:

equality constraint $ψ(x_T, θ) + \int_{0}^{T} L(t,x,u,θ) dt = 0$,

and/or (depending on the problem it self)

inequality constraint $ψ(x_T, θ) + \int_{0}^{T} L(t,x,u,θ) dt \leq 0$.

User can define and registe following six type of constraints to the package and they will be transformed in to above canonical constraints.

- Terminal equality: $ψ(x_T, θ)=0$
- Terminal inequality: $φ(x_T, θ) ≤ 0$
- Integral equality: $∫ q(t,x,u,θ) dt = b$
- Integral inequality: $∫ q(t,x,u,θ) dt ≤ b$
- Path inequality: $h(t,x,u,θ) ≤ b$
- Path equality: $h(t,x,u,θ) = 0$

> ℹ️ **Note** Path constraints are transformed into integral form by applying the constraints transcription techniques.

> 😊 Full template for all supported constraints `my_A_problem_temeplate.py` can be found in `admiser\templates`.
## Free Terminal Time
The free-terminal time problem is a special type of optimal control problem, where the objective function is typically to minimize the system's runtime.

Currently, users need to manually transform this type of problem into a "standard" optimal control problem before defining and solving it. The transformation process is as follows:

$$
\begin{aligned}
\min\;& t_f \\
\text{s.t.}\;& \frac{dx}{dt} = f(t,x,u,\theta)
\end{aligned}
$$
Introduce new state-control pire ($x_{extra}$ and $u_{extra}$) for the shortest time decision variable $t_f$:
$$
\begin{aligned}
\min\;& x_{\text{extra}} \\
\text{s.t.}\;& \frac{dx}{dt} = f(t,x,u,\theta)\,x_{\text{extra}} \\
& \frac{d x_{\text{extra}}}{dt} = u_{\text{extra}}
\end{aligned}
$$

> 😊 Examples can be found in `admiser\examples\my_free_terminal_time.py` and `admiser\examples\my_free_terminal_time2.py`.


## Cite / Acknowledge

Theoretical Framework for control parametrization originally proposed by Professor Kok Lay Teo and other co-researchers can be found in:

**Goh, C.J., & Teo, K.L.** (1988). *Control parametrization: a unifiedapproach to optimal control problems with general constraints*.**Automatica**, 24(1), 3–18.  
  
**Goh, C.J., & Teo, K.L.** (1991). *Alternative algorithms for solvingnonlinear function and functional inequalities*. **Applied Mathematics andComputation**, 41(2), 159–177.  

For the latest theoretical and computational methods based on control parameterization technology, please refer to the following monograph:
   
**Teo, K.L., Li, B., Yu, C., & Rehbock, V.** (2021). *Applied and Computational Optimal Control: A Control Parametrization Approach*. Springer Optimization and Its Applications (Vol. 171). Springer International Publishing. https://doi.org/10.1007/978-3-030-69913-0

The gradient process is facilitated by using the automatic differentiation technique to replace the continuous adjoint in Teo's original method. AD is achieved by using the following off-the-shelf tools.

**Bell, B.** (2007). *CppAD: A package for C++ algorithmic differentiation*. http://www.coin-or.org/CppAD
