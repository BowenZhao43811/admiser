# ADMISER — Automatic Differentiation Ehanced Modern Control-Parametrization OCP Solver

ADMISER is a **Numerical Optimal Control** toolkit that incorporates **Automatic Differentiation** for gradient computation within [Professor Kok Lay Teo](https://sunwayuniversity.edu.my/school-of-mathematical-sciences/staff-profiles/professor-teo-kok-lay)'s **Control Parametrization** framework in Python.


- ✅ **Control Parametrization** (piecewise-constant controls; policy/feedback parametrization)
- ✅ **Constraint Transcription** (smooth path inequalities → canonical integral constraints), with an ε→0 **continuation loop**
- ✅ **Canonical constraints**: terminal eq/ineq, integral eq/ineq, path eq/ineq (smoothed)
- ✅ **Multi-substep RK4** integrator with **4th-order** substep quadrature (cost integrals are as accurate as the state trajectory, at no extra dynamics evaluations)
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
- [Choosing ε and γ](#choosing-ε-and-γ)
- [Quadrature Accuracy](#quadrature-accuracy)
- [Writing AD-safe Model Functions](#writing-ad-safe-model-functions)
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
pip install --upgrade pip wheel setuptools
```

### install `ADMISER`
From the repository root:
```sh
pip install "admiser @ git+https://github.com/BowenZhao43811/admiser.git"
```
> ⚠️ **Python version requirement**: Python ≥ 3.10 (enforced by `requires-python`)

> ⚠️**Dependencies** `numpy`, `scipy`, `matplotlib`, `cppad_py` (Auto installations are provided in the next subsection).

### install cppad and build cppad_py
```sh
admiser-install-cppad
```
> ⚠️ Minimum requirements for running the package are provided, for details on CppAD and CppAD_Py please refer to COIN's official documentations.

## Quickstart

The jupyter notebook document `main_solver_entrence.ipynb` distributed together with the package records the solution results of many examples from different industries. Definitions of these example problems can be found in the `admiser.examples.ProblemName.py`

### find the accompanying example solution 
```py
from admiser.examples import get_notebook_path

path = get_notebook_path()
print("Notebook path:", path)
```
Grab the file path then open it with your prefered editor (don't forget to select your venv where admiser was installed).

## Core Ideas

- **Control Parametrization**: treat controls as decision variables over segments; states propagate by integrating dynamics.

- **Constraint Transcription**: 
    - *Path inequality* $h(t) ≤ 0$ → smooth hinge $L_ε(h)$; enforce $∫ L_ε(h) dt ≤ γ$ with $γ = Tε/4$ (see [Choosing ε and γ](#choosing-ε-and-γ)).
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
objective_builder = make_builders(dyn=dyn, L=L, Phi=None, quad='rk4')

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

res = OCPSolver(problem).solve(maxiter=800, ftol=1e-9)
print(res["J_opt"])
```

> ℹ️ **`solve()` is the only solve entry point.** Whether the path-constraint transcription is solved once or by an ε→0 continuation is declared *in the problem*, with `problem.set_transcription(...)` — see [Choosing ε and γ](#choosing-ε-and-γ). The result dict has the same shape either way.
>
> If you want the transcribed NLP *without* solving it — to check the AD gradient against finite differences, say — use `to_nlp()`:
> ```py
> nlp = OCPSolver(problem).to_nlp()      # records the AD tape, runs no optimizer
> g_ad = nlp.objective_grad(z)
> ```

> ℹ️ **Figure Output**: States and control trajectories ploting is not integrated into the solver, instead, the flaxibility is given to the users. User can access raw optimization reaults of control, system parameters, objective, state, residual on the cononical equality and inequalities by 
`res["U_opt"]`
`res.get("theta_opt", None)`
`res["J_opt"]`
`res["X_opt"]`
`res["t_opt"]`
`res.get("eq_resid", None)`
`res.get("ineq_resid", None)`
`res.get("path_viol", None)`.

> ℹ️ `eq_resid` should be ≈ 0, `ineq_resid` is $C(z) \ge 0$, and `path_viol` gives $\max_t h(t)$ on the grid for each registered path inequality — the direct measure of how well the transcription held (should be ≤ 0).

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

## Choosing ε and γ

For a path inequality $h(t) \le 0$, `add_path_ineq(hfun, eps, gamma=None)` enforces

$$\int_0^T L_ε(h(t))\,dt \le γ .$$

**Leave `gamma` unset.** It then defaults to $γ = Tε/4$, which is the value compatible with $ε$:

- $L_ε(h) - \max(h,0) \in [0, ε/4]$, with the maximum attained at $h=0$. So any trajectory that genuinely satisfies $h(t)\le 0$ also satisfies $\int L_ε \,dt \le Tε/4$ — the transcribed problem does **not** exclude the true optimum.
- Conversely $\int \max(h,0)\,dt \le \int L_ε\,dt \le γ = Tε/4$, so the $L^1$ violation is bounded by $Tε/4$ and vanishes as $ε \to 0$.

> ⚠️ Do **not** use `gamma=0`. Since $L_ε \ge 0$, requiring $\int L_ε\,dt \le 0$ forces $L_ε \equiv 0$, i.e. $h(t) \le -ε$ everywhere — a strictly interior solution, and one that routinely makes SLSQP stall on the constraint boundary.

> ⚠️ **ε carries the units of `h`.** If $h$ ranges over $10^5$, then `eps=1e-3` smooths the hinge over a relative width of $10^{-9}$ — effectively a nondifferentiable hinge. Scale `eps` to the magnitude of your own $h$ (see `admiser/examples/my_medical1.py`).

### ε-continuation

Small ε gives a tight approximation but a nearly nonsmooth NLP; large ε is smooth but loose. The standard remedy is to start large and shrink, warm-starting each solve. Declare it **in the problem definition** — the solving side never has to know:

```py
# in my_problem.py, next to the constraint it governs
problem.add_path_ineq(hfun=hfun, eps=1e-4)      # eps is the *final* value; gamma auto = Teps/4
problem.set_transcription(mode="continuation", n_rounds=4, shrink=0.1)
```

```py
# in the driver — unchanged, whichever mode the problem declared
res = OCPSolver(problem).solve(maxiter=1000, ftol=1e-12)

for r in res["rounds"]:
    print(r["eps"], r["gamma"], r["J_opt"], r["max_path_viol"], r["status"])
```

```
[ADMISER] round 1/4 eps=1.000e-01  J=+0.14636838  max h(t)=+7.975e-02  status=0  nit=95
[ADMISER] round 2/4 eps=1.000e-02  J=+0.167629    max h(t)=+7.649e-03  status=0  nit=106
[ADMISER] round 3/4 eps=1.000e-03  J=+0.1700801   max h(t)=+1.144e-05  status=0  nit=131
[ADMISER] round 4/4 eps=1.000e-04  J=+0.17042904  max h(t)=-5.965e-04  status=0  nit=107
```

The starting ε is given as a **ratio**, not an absolute value: ε carries the units of its own `h`, so several path constraints cannot share one absolute start — but they can share a shrink ratio. Each constraint runs `eps/shrink^(n_rounds-1) → … → eps`, landing exactly on its registered value in the final round.

`mode="single"` (the default, and what you get if you never call `set_transcription`) is simply the one-round case, so both modes go through the same code path and return the same result shape — `rounds` just has length 1. `gamma` is re-derived as $Tε/4$ each round for constraints registered without an explicit `gamma`; an explicitly given `gamma` stays fixed. `eps` is baked into the tape as a constant, so each round re-records the tape (cheap relative to the SQP iterations).

`solve()` never mutates the problem: ε is only scaled inside the round and restored afterwards, so the same `problem` can be solved repeatedly with identical results.

## Quadrature Accuracy

The **state** is always advanced by full RK4. The integrals along that trajectory — the objective $\int L\,dt$, every integral constraint, and every transcribed path constraint — are accumulated by a separately chosen **quadrature scheme**, and its order is *not* automatically the same as the integrator's.

Set it with `make_builders(quad=...)`, or globally with `problem.quad_scheme` (which takes precedence, and applies to the objective **and** every constraint — they must share one scheme, or the KKT point is meaningless).

| `quad` | order | `n_eval` | sampling |
|---|---|---|---|
| `'rk4'` *(default)* | **4** | 4 | the four RK4 stage points, weights $h/6, h/3, h/3, h/6$ |
| `'simpson'` | 3 | 3 | Simpson with midpoint $x + \tfrac{h}{4}(k_1+k_2)$ |
| `'midpoint'` | 2 | 1 | Euler half-step midpoint $x + \tfrac{h}{2}k_1$ |
| `'trapezoid'` | 2 | 2 | both endpoints, weights $h/2, h/2$ |
| `'left'` / `'right'` | 1 | 1 | substep left / right endpoint |

`QUAD_SCHEMES` is the single source of scheme names — there are no aliases. A misspelled or outdated name raises immediately rather than being silently accepted as a different accuracy.

Every scheme reuses the $k_1..k_4$ that RK4 already computed, so **none of them cost extra evaluations of `dyn`**. `n_eval` is the number of *integrand* evaluations per substep — it is what drives AD tape size, so the price of a higher order is a bigger tape, not a second ODE solve.

Measured orders on $\dot x = x,\ g = t^2 x,\ \int_0^1 t^2 e^t\,dt = e-2$:

```
scheme      n_eval      m=1       m=2       m=4       m=8      m=16   order
left            1   2.97e-01  1.59e-01  8.23e-02  4.18e-02  2.11e-02   0.98
right           1   3.82e-01  1.80e-01  8.76e-02  4.31e-02  2.14e-02   1.02
midpoint        1   2.61e-02  6.64e-03  1.67e-03  4.19e-04  1.05e-04   2.00
trapezoid       2   4.23e-02  1.06e-02  2.65e-03  6.64e-04  1.66e-04   2.00
simpson         3   1.12e-04  1.70e-05  2.29e-06  2.95e-07  3.75e-08   2.96
rk4             4   5.12e-05  3.22e-06  2.01e-07  1.26e-08  7.84e-10   4.00
```

### Why the node accuracy matters more than the rule

A scheme's order is capped by **both** the quadrature rule and the accuracy of the state estimate at each node — and the second is usually the binding one:

- Simpson's rule is 4th order, but with the Euler half-step midpoint $x+\tfrac{h}{2}k_1$ (error $O(h^2)$) the whole thing collapses to **order 2** — three integrand evaluations buying nothing over `midpoint`'s one. It reaches order 3 only with the *averaged* midpoint $x+\tfrac{h}{4}(k_1+k_2)$, whose leading errors $\mp\tfrac{1}{8}h^2 f'f$ cancel.
- `'rk4'`'s fourth node must be the stage-4 argument $x + h k_3$, **not** the step-end state $x_{n+1}$. Substituting $x_{n+1}$ breaks RK4's order conditions and drops it to order 3 (measured 2.96).

`'rk4'` is exactly RK4 applied to the augmented state $\dot y = L(t,x,u,θ)$, which is why $\int L\,dt$ comes out as accurate as the trajectory.

Programmatic access: `admiser.QUAD_SCHEMES` (name → order / `n_eval` / summary), `admiser.quad_order(name)`, `admiser.validate_quad_scheme(name)`.

## Writing AD-safe Model Functions

Your `dyn`, `L`, `Phi`, `qfun` and `hfun` are executed **once**, on `a_double` values, to record the tape. Anything that is not a recorded CppAD operation gets frozen at the recording point:

```py
def dyn(x, u, theta=None):
    if u[0] > 0:                 # ❌ this branch is decided once, at the taping point,
        return np.array([u[0]], dtype=object)   #    and silently reused for every z
    return np.array([-u[0]], dtype=object)
```

- ❌ Python `if` / `min` / `max` / `abs` / `np.where` on AD values, and `float(...)` casts of them.
- ✅ `np.exp/sin/cos/sqrt/**` on `a_double` (they dispatch to CppAD operations), and `admiser.L_eps` / `admiser.smooth_abs`, which use `cond_assign` so the comparison is recorded as a real `CondExp` operator and re-evaluated on every pass.

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
