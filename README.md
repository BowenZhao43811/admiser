# ADMISER — Automatic Differentiation Enhanced Modern Control-Parametrization OCP Solver

ADMISER is a **Numerical Optimal Control** toolkit that incorporates **Automatic Differentiation** for gradient computation within [Professor Kok Lay Teo](https://sunwayuniversity.edu.my/school-of-mathematical-sciences/staff-profiles/professor-teo-kok-lay)'s **Control Parametrization** framework in Python.


- ✅ **Control Parametrization** (piecewise-constant controls; policy/feedback parametrization)
- ✅ **Time Scaling (CPET)**: segment durations become decision variables, so switching instants and free terminal times are found exactly instead of being quantised to a fixed grid
- ✅ **Constraint Transcription** (smooth path inequalities → canonical integral constraints), with an ε→0 **continuation loop** that tightens the approximation round by round, from inexact to exact
- ✅ **Canonical constraints**: terminal eq/ineq, integral eq/ineq, path eq/ineq (smoothed)
- ✅ **Multi-substep RK4** integrator with **4th-order** substep quadrature (cost integrals are as accurate as the state trajectory, at no extra dynamics evaluations)
- ✅ **System Parameters** are simultaneously optimized within the same framework (if have)
- ✅ **Automatic Differentiation** via [JAX](https://github.com/jax-ml/jax), in reverse mode: one compiled function delivers the objective, every constraint and all their derivatives
- ✅ **Automatic Scaling**: objective and constraints are normalised internally so SLSQP's absolute `ftol` behaves as the relative tolerance users expect; every reported number comes back in your own units
- ✅ **SQP** method for solving nonlinear programming problem

> ✅ **Runs natively on Windows and Linux.** Everything installs with `pip` — no compiler, no system libraries, no WSL.

---

## Table of Contents
- [Install](#install)
- [Quickstart](#quickstart)
- [Core Ideas](#core-ideas)
- [Problem Template](#problem-template)
- [Registering Constraints](#registering-constraints)
- [Choosing ε and γ](#choosing-ε-and-γ)
- [Solve Status](#solve-status)
- [Automatic Scaling](#automatic-scaling)
- [Quadrature Accuracy](#quadrature-accuracy)
- [Writing Model Functions for JAX](#writing-model-functions-for-jax)
- [Time Scaling (CPET)](#time-scaling-cpet)
- [Free Terminal Time](#free-terminal-time)
- [Cite / Acknowledge](#cite--acknowledge)

---

## Install

ADMISER is pure Python on top of NumPy, SciPy and JAX, and all of them install
with `pip`. It runs natively on **Windows** and on **Linux**; no compiler is needed.

### 1. create a virtual environment
A venv keeps ADMISER and its dependencies out of your global Python.

Windows (PowerShell):
```powershell
python -m venv admiser_venv
admiser_venv\Scripts\activate
```

Linux:
```sh
python3 -m venv ~/admiser_venv
source ~/admiser_venv/bin/activate
```

### 2. install `ADMISER`
```sh
pip install "admiser @ git+https://github.com/BowenZhao43811/admiser.git"
```

Or, from a clone of the repository — editable, together with the test tools — and
check that everything works:
```sh
pip install -e ".[dev]"
pytest
```

> ⚠️ **Python version requirement**: Python ≥ 3.10 (enforced by `requires-python`). `pip` installs the newest JAX your Python supports; JAX 0.11 needs Python ≥ 3.12, and on 3.10 / 3.11 `pip` falls back to an older JAX by itself.

> ⚠️ **Dependencies** `numpy`, `scipy`, `matplotlib`, `jax` are installed automatically.

> ℹ️ On a machine with an NVIDIA GPU, JAX may print *"An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu."* That is harmless: ADMISER's problems are small and dense, which is exactly what the CPU is good at.

## Quickstart

The Jupyter notebook `main_solver_entrence.ipynb` distributed together with the package records the solution results of many examples from different industries. Definitions of these example problems can be found in the `admiser.examples.ProblemName.py`

### find the accompanying example solution 
```py
from admiser.examples import get_notebook_path

path = get_notebook_path()
print("Notebook path:", path)
```
Grab the file path, then open it with your preferred editor (remember to select the venv where admiser is installed).

## Core Ideas

- **Control Parametrization**: treat controls as decision variables over segments; states propagate by integrating dynamics.

- **Constraint Transcription**: 
    - *Path inequality* $h(t) ≤ 0$ → smooth hinge $L_ε(h)$; enforce $∫ L_ε(h) dt ≤ γ$ with $γ = Tε/4$ (see [Choosing ε and γ](#choosing-ε-and-γ)).
    - *Path equality* $h(t)=0$ → $∫ h^2 dt = 0$.

- **AD with JAX**: the whole problem — one RK4 rollout that feeds the objective $G_0(z)$ and every constraint $G_i(z)$ — is a single `jax.numpy` function. JAX compiles it **once** and differentiates it in **reverse mode**, whose cost grows with the number of outputs (one objective plus a handful of constraints), not with the number of control values. That is what makes reverse mode the right choice for control parametrization.

## Problem Template

### Problem define

Create a problem file (e.g., `my_problem.py`) using the following simple template pattern (😊 A full template covering every supported constraint, `my_A_problem_template.py`, is in `admiser/templates`):
```py
import numpy as np
import jax.numpy as jnp            # model functions are written with jax.numpy
from admiser import OCPProblem
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
    return jnp.array([x2, -x1 + u[0]])

def L(t, x, u, theta):
    return x[0]*x[0] + x[1]*x[1] + 0.25*(u[0]*u[0])

# No terminal cost, no terminal equality, no integral equalities
objective_builder = make_builders(dyn=dyn, L=L, Phi=None, quad='rk4')

# Control bounds
def control_bounds_builder(p): return [(-10.0, 10.0)] * (p.N * p.nu)

# Assemble
problem = OCPProblem(
    N=N, dt=dt, x0=x0, u0=u0, dyn=dyn,
    m_sub=10,                      # RK4 substeps per control segment
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
print(res["status"], res["message"])   # 0 means converged; see "Solve Status"
print(res["J_opt"])
```

### The public API

The surface is deliberately small — almost every problem needs only these:

```py
from admiser import OCPProblem, OCPSolver, make_builders
```

| name | what it is |
|---|---|
| `OCPProblem` | the problem, and the policies for how it should be solved |
| `OCPSolver` | `solve()` and `to_nlp()` |
| `make_builders` | declares the objective $\int L\,dt + \Phi$ |
| `QUAD_SCHEMES` | the quadrature schemes, their orders and costs |
| `L_eps`, `smooth_abs` | JAX-safe smoothing helpers, for use inside your own model functions |

Everything else lives in a named module and can be imported from there:

| module | contents |
|---|---|
| `admiser.problem_definition` | `OCPProblem` |
| `admiser.objective_builder` | `make_builders` |
| `admiser.constraint_smoothing` | `L_eps`, `smooth_abs` |
| `admiser.quadrature` | `rk4_substep`, `rk4_substeps`, `rk4_step`, `QUAD_SCHEMES`, `quad_order` |
| `admiser.ocp_to_nlp` | `build_nlp` — turns the whole problem into compiled NLP functions (JAX) |
| `admiser.nlp_functions` | `NLPFunctions` — those functions presented to SciPy |
| `admiser.problem_scaling` | `compute_scaling`, `ProblemScaling` |
| `admiser.sqp_solver` | `OCPSolver`, and the `STATUS_*` codes of [Solve Status](#solve-status) |

> ℹ️ **`solve()` is the only solve entry point.** Whether the path-constraint transcription is solved once or by an ε→0 continuation is declared *in the problem*, with `problem.set_transcription(...)` — see [Choosing ε and γ](#choosing-ε-and-γ). The result dict has the same shape either way.
>
> If you want the transcribed NLP *without* solving it — to check the AD gradient against finite differences, say — use `to_nlp()`:
> ```py
> nlp = OCPSolver(problem).to_nlp()      # builds the NLP, runs no optimizer
> g_ad = nlp.objective_grad(z)
> ```

> ℹ️ **Figure Output**: Plotting of states and control trajectories is deliberately left out of the solver, so you keep full control over presentation. The raw optimisation results -- how the solve ended, controls, system parameters, objective, states, and the residuals of the canonical equalities and inequalities -- are available as 
`res["status"]`
`res["message"]`
`res["U_opt"]`
`res.get("theta_opt", None)`
`res["J_opt"]`
`res["X_opt"]`
`res["t_opt"]`
`res.get("eq_resid", None)`
`res.get("ineq_resid", None)`
`res.get("path_viol", None)`.

> ℹ️ `eq_resid` should be ≈ 0, `ineq_resid` is $C(z) \ge 0$, and `path_viol` gives $\max_t h(t)$ on the grid for each registered path inequality — the direct measure of how well the transcription held (should be ≤ 0).

> 😊 A ready-made plotting driver, `solve_A_my_problem_template.py`, is available in `admiser/templates`


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

> 😊 A full template covering every supported constraint, `my_A_problem_template.py`, is in `admiser/templates`.

## Choosing ε and γ

For a path inequality $h(t) \le 0$, `add_path_ineq(hfun, eps, gamma=None)` enforces

$$\int_0^T L_ε(h(t))\,dt \le γ .$$

**Leave `gamma` unset.** It then defaults to $γ = Tε/4$, which is the value compatible with $ε$:

- $L_ε(h) - \max(h,0) \in [0, ε/4]$, with the maximum attained at $h=0$. So any trajectory that genuinely satisfies $h(t)\le 0$ also satisfies $\int L_ε \,dt \le Tε/4$ — the transcribed problem does **not** exclude the true optimum.
- Conversely $\int \max(h,0)\,dt \le \int L_ε\,dt \le γ = Tε/4$, so the $L^1$ violation is bounded by $Tε/4$ and vanishes as $ε \to 0$.

> ⚠️ Do **not** use `gamma=0`. Since $L_ε \ge 0$, requiring $\int L_ε\,dt \le 0$ forces $L_ε \equiv 0$, i.e. $h(t) \le -ε$ everywhere — a strictly interior solution, and one that routinely makes SLSQP stall on the constraint boundary.

> ⚠️ **ε carries the units of `h`.** If $h$ ranges over $10^5$, then `eps=1e-3` smooths the hinge over a relative width of $10^{-9}$ — effectively a nondifferentiable hinge. Scale `eps` to the magnitude of your own $h$ (see `admiser/examples/my_medical1.py`).

### ε-continuation

Small ε gives a tight approximation but a nearly nonsmooth NLP; large ε is smooth but loose. The standard remedy is to start large and shrink, warm-starting each solve from the previous one — the approximation goes from **inexact to exact**. Declare it **in the problem definition**; the solving side never has to know:

```py
# in my_problem.py, next to the constraint it governs
problem.add_path_ineq(hfun=hfun, eps=1e-1)      # eps is the STARTING value; gamma auto = T*eps/4
problem.set_transcription(mode="continuation", n_rounds=4, shrink=0.1)
```

```py
# in the driver — unchanged, whichever mode the problem declared
res = OCPSolver(problem).solve(maxiter=1000, ftol=1e-12)

for r in res["rounds"]:
    print(r["eps"], r["gamma"], r["J_opt"], r["max_path_viol"], r["scipy_status"])
```

```
[ADMISER] round 1/4  eps=1.000e-01  J=+0.14636838  max h(t)=+7.975e-02  SLSQP status=0  nit=89
[ADMISER] round 2/4  eps=1.000e-02  J=+0.167629  max h(t)=+7.646e-03  SLSQP status=0  nit=98
[ADMISER] round 3/4  eps=1.000e-03  J=+0.1700801  max h(t)=+1.136e-05  SLSQP status=0  nit=124
[ADMISER] round 4/4  eps=1.000e-04  J=+0.17042904  max h(t)=-5.966e-04  SLSQP status=0  nit=106
[ADMISER] status 0: every round converged (returning round 4/4)
```

The rounds run `eps → eps·shrink → eps·shrink² → …`, `n_rounds` of them, each warm-started from the one before. `gamma` shrinks along with `eps`: a `gamma` left automatic is re-derived as $Tε/4$ each round, and an explicitly given `gamma` is multiplied by the same `shrink` factor as `eps`. Each path constraint keeps its own `eps` — ε carries the units of its own `h` — and they all share the shrink ratio.

**The continuation stops at the first round that fails** — SLSQP has then reached the smallest ε it can still handle — and returns the last round that converged. `res["status"]` says why it stopped; see [Solve Status](#solve-status). `res["final_eps"]` and `res["final_gamma"]` say where the returned solution was solved.

`mode="single"` (the default, and what you get if you never call `set_transcription`) is simply the one-round case: it solves once at the registered `eps`. Both modes go through the same code path and return the same result shape — `rounds` just has length 1.

ε and γ are passed into the compiled NLP as **arguments**, so all rounds share one compilation, and `solve()` never mutates the problem: the same `problem` can be solved repeatedly with identical results.

## Solve Status

`res["status"]` is ADMISER's own code for how the solve as a whole ended — every round of a continuation, or the single round of a plain solve:

| `status` | meaning | what is returned |
|---|---|---|
| **0** | every round converged | the last round |
| **1** | a round hit the SLSQP iteration limit (`maxiter`) | that round's iterate — usually far better than its starting point, but **not converged** |
| **2** | a later round failed for another reason | the last round that converged |
| **3** | the first round already failed | its iterate anyway, since there is nothing better — **not converged** |

`res["message"]` is the readable form, and `res["success"]` is `status == 0`. Anything other than 0 also raises a `RuntimeWarning`, so a result that did not converge is never returned silently. The codes are defined in `admiser.sqp_solver` as `STATUS_CONVERGED`, `STATUS_MAX_ITERATIONS`, `STATUS_SLSQP_FAILURE` and `STATUS_NO_ROUND_SUCCEEDED`.

SciPy's own SLSQP status of every round is kept in `res["rounds"][k]["scipy_status"]`, with its `message` and `nit`. The ones you will meet most often: `0` converged, `4` inequality constraints incompatible (infeasible), `8` positive directional derivative in the line search (often a scaling problem), `9` iteration limit reached.

> ⚠️ Status 0 is necessary, not sufficient. Also look at `eq_resid`, `ineq_resid` and `path_viol` before trusting a result.

## Automatic Scaling

**On by default.** Write your objective and constraints in whatever units are
natural for the problem; the solver normalises them internally and converts
everything it reports back into your units.

### Why

SciPy's SLSQP compares the **absolute** change in the objective against `ftol`.
With an objective of order $10^6$, `ftol=1e-9` demands a relative accuracy of
$10^{-15}$ — below machine precision — so the line search reports "no descent
direction" after a handful of iterations and stops, possibly at an **infeasible**
point, with no error raised.

That was not hypothetical. `my_medical1` has an objective of order $2.5\times10^6$
and used to halt after 9 iterations with its dose budget violated by $10^{-2}$.
Sweeping `ftol` from `1e-12` to `1e0` changed nothing — the tolerance was simply
inoperative.

Multiplying the objective by a positive constant cannot move the minimiser, but
it does change what `ftol` means. **Normalising turns SLSQP's absolute test into
an effectively relative one** — which is what most people assume `ftol` already is.

### Two rules, one idea

The objective and the constraints are deliberately **not** scaled by the same
formula:

| | scaled by | measured from | fixes |
|---|---|---|---|
| objective | one global factor $1/\lvert J\rvert$ | the **value** at the initial guess | `ftol` being an absolute test |
| constraints | one factor **per row** $1/\lVert\nabla g_i\rVert_\infty$ | the **Jacobian** | conditioning of the QP subproblem |

Normalising a constraint by its *value* would be wrong: a constraint that happens
to be satisfied at the starting point has $g_i \approx 0$ there, and that is
normal, even desirable.

### Usage

```py
problem.set_scaling(objective="auto", constraints="auto")   # the default
problem.set_scaling(objective="none", constraints="none")   # switch it off
problem.set_scaling(objective=1e-6)                         # pick the factor yourself
```

`solve()` prints one line stating what it did, and `res["scaling"]` carries the
factors, so the transform is never invisible:

```
[ADMISER] scaling applied: objective x3.900e-07 (auto); constraints 1/2 rows scaled, factors in [1.00e+00, 4.00e+00] (auto)
[ADMISER]   all reported values are converted back to your units
```

### What it changed

All thirteen examples now converge with `status=0` at a feasible point. The two
that previously did not:

| example | before | after |
|---|---|---|
| `my_medical1` | status 8, 9 iterations, budget violated by $10^{-2}$ | status 0, feasible to $6\times10^{-14}$ |
| `my_medical2` | status 8, violation $6\times10^{-7}$ | status 0, feasible to $1\times10^{-13}$ |

Neither example carries a hand-written scale factor — both state their objectives
in natural units.

Constraint scaling on its own is a smaller effect: measured across all examples it
saves about 8% of the total iterations, with one problem
(`my_free_terminal_time2`) taking half as many and none becoming less feasible.

> ℹ️ The compiled NLP always computes **your** problem, unscaled. Scaling is
> applied only at the boundary with SciPy, so `to_nlp()` hands back your own
> problem — a gradient checked there against finite differences is the gradient
> of what you wrote.

> ℹ️ The factors are estimated from **fixed** probe offsets around the initial
> guess, never random ones, so the same problem always produces the same factors
> and two runs are comparable.

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

Every scheme reuses the $k_1..k_4$ that RK4 already computed, so **none of them cost extra evaluations of `dyn`**. `n_eval` is the number of *integrand* evaluations per substep, so the price of a higher order is a few more integrand evaluations, not a second ODE solve.

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

Programmatic access: `admiser.QUAD_SCHEMES` maps each name to its order, `n_eval` and a one-line summary. `quad_order(name)` and `validate_quad_scheme(name)` live in `admiser.quadrature`.

## Writing Model Functions for JAX

While the solver builds the NLP, your `dyn`, `L`, `Phi`, `qfun`, `hfun`, terminal functions and `x0_from_theta` are not called on plain numbers. JAX calls them with **tracers**: stand-ins that record every operation, so that the whole computation can be compiled and differentiated. Two rules follow.

**1. Use `jax.numpy`, not `numpy`.**

- ✅ `jnp.array([dx1, dx2])`, `jnp.exp`, `jnp.sin`, `jnp.sqrt`, `**` and ordinary arithmetic
- ❌ `np.array([...], dtype=object)`, `np.exp(...)` — NumPy cannot look inside a tracer

**2. Decide with `jnp.where`, not with `if`.**

```py
def dyn(x, u, theta=None):
    if u[0] > 0:                          # ❌ a tracer has no True/False value yet
        return jnp.array([u[0]])
    return jnp.array([-u[0]])

def dyn(x, u, theta=None):                # ✅ both branches are recorded, and the
    return jnp.array([jnp.where(u[0] > 0, u[0], -u[0])])   # right one is picked every time
```

For the same reason, do not call `float(...)` or `int(...)` on a state, a control or a parameter. Branching on things that are fixed — `N`, a flag, `theta is None` — is fine.

Breaking a rule raises a `TypeError` straight away, before any optimisation starts, with these hints attached. (Under the CppAD backend of earlier versions, the same `if` was silently frozen into the tape and produced wrong derivatives without any error.)

`admiser.L_eps` and `admiser.smooth_abs` follow the rules, so you can use them inside your own functions. Your functions still accept plain NumPy arrays too, so you can call them directly to evaluate or plot a constraint.

> ℹ️ Importing `admiser` switches JAX to 64-bit floats for the whole Python process, because SLSQP needs the precision. Other JAX code running in the same session will compute in 64-bit as well.

## Time Scaling (CPET)

Without the transform the knot points are fixed and uniform: the optimiser
chooses only the control **value** on each segment, never where the segments end.
For a bang-bang solution the switching instant can then only land on a grid
point, so resolving it costs a fine grid — and because the grid is uniform, that
cost is paid everywhere, not just near the switch.

Teo's **Control Parameterization Enhancing Transform** makes the segment
durations decision variables. Introduce a new independent variable $s$ on which
the grid *is* uniform, and let real time flow at an optimisable rate:

$$\frac{dt}{ds} = \tau_k, \qquad s \in [k-1,\ k)$$

Segment $k$, of width 1 in $s$, therefore lasts $\tau_k$ in real time. The
augmented system is

$$\frac{dx}{ds} = \tau_k\, f(t,x,u_k), \qquad \frac{dt}{ds} = \tau_k$$

so the ODE is still solved on a fixed uniform grid, while all the awkward
variability has been absorbed into ordinary decision variables. The decision
vector becomes $z = [U\ (N n_u)\ ;\ \theta\ (n_\theta)\ ;\ \tau\ (N)]$.

Enable it in the problem definition:

```py
problem.set_time_scaling(tau0=dt, tau_min=0.0, tau_max=None, total_time=None)
```

| argument | meaning |
|---|---|
| `tau0` | initial guess for the durations; `None` gives a uniform split, $\tau_k = dt$ |
| `tau_min` | lower bound, default `0.0`. Zero is the standard choice: a duration collapsing to zero is how the optimiser **drops a segment it does not need**. Negative durations would run time backwards and are always rejected |
| `tau_max` | upper bound, `None` for unbounded |
| `total_time` | pin the horizon with $\sum \tau_k = T$. Leave it `None` for a **free** horizon |

`total_time=T` is nothing special: it registers exactly
`add_integral_eq(qfun=lambda t, x, u, th: 1.0, target=T)`, because
$\int_0^T 1\,dt = \sum_k \tau_k$.

The solve entry point does not change — `OCPSolver(problem).solve(...)` — and the
result dict gains `tau_opt` (the optimised durations) and `T_opt` (their sum).
`t_opt` becomes the true, non-uniform knot grid.

### What it buys: resolving a switch

On `my_bang_bang`, where the optimal control switches once:

```
setup                               n_vars              J*
fixed grid, N=4                          8   -41.250000000
fixed grid, N=8                         16   -41.250000000
fixed grid, N=21                        42   -41.301155383
fixed grid, N=84                       168   -41.348396501
CPET,       N=4                         12   -41.309894531
CPET,       N=8                         24   -41.349001795
```

CPET with 8 segments and 24 variables beats a uniform grid of 84 segments and 168
variables. On a fixed grid the switch is quantised to the knot spacing, and the
only way to sharpen it is to refine everywhere.

## Free Terminal Time

A minimum-time problem

$$\min\ t_f \quad \text{s.t.}\quad \frac{dx}{dt} = f(t,x,u,\theta)$$

needs no rewriting. Turn the transform on, leave `total_time` unset so the
horizon is free, and make the objective the elapsed time:

```py
def L(t, x, u, theta):
    return 1.0                       # int 1 dt == sum(tau) == the horizon

problem.set_time_scaling(tau0=dt, tau_min=1e-4)
```

```py
res = OCPSolver(problem).solve()
print(res["T_opt"])                  # the minimum time
print(res["tau_opt"])                # where the segments ended up
```

> ℹ️ **Earlier versions required a manual rewrite** — introducing an extra state
> $x_\text{extra}$ and control $u_\text{extra}$ so that every equation read
> $\dot x = f(t,x,u,\theta)\,x_\text{extra}$ with $\dot x_\text{extra} = u_\text{extra}$,
> then minimising $x_\text{extra}(T)$. That *is* the time-scaling transform, written
> out by hand. It is no longer necessary: state and control keep their natural
> meaning and dimensions.

> 😊 Both `admiser/examples/my_free_terminal_time.py` and
> `admiser/examples/my_free_terminal_time2.py` have been rewritten this way. They
> reproduce the minimum times of the hand-transformed versions (4.3211735 and
> 46.17131), and the second one now converges cleanly where the manual version
> used to stop with SLSQP status 8.

## Cite / Acknowledge

Theoretical Framework for control parametrization originally proposed by Professor Kok Lay Teo and other co-researchers can be found in:

**Goh, C.J., & Teo, K.L.** (1988). *Control parametrization: a unified approach to optimal control problems with general constraints*.**Automatica**, 24(1), 3–18.  
  
**Goh, C.J., & Teo, K.L.** (1991). *Alternative algorithms for solving nonlinear function and functional inequalities*. **Applied Mathematics and Computation**, 41(2), 159–177.  

For the latest theoretical and computational methods based on control parameterization technology, please refer to the following monograph:
   
**Teo, K.L., Li, B., Yu, C., & Rehbock, V.** (2021). *Applied and Computational Optimal Control: A Control Parametrization Approach*. Springer Optimization and Its Applications (Vol. 171). Springer International Publishing. https://doi.org/10.1007/978-3-030-69913-0

The gradient process is facilitated by using the automatic differentiation technique to replace the continuous adjoint in Teo's original method. AD is achieved by using the following off-the-shelf tool.

**Bradbury, J., Frostig, R., Hawkins, P., Johnson, M.J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q.** (2018). *JAX: composable transformations of Python+NumPy programs*. http://github.com/jax-ml/jax

Earlier versions of ADMISER used CppAD for the same purpose:

**Bell, B.** (2007). *CppAD: A package for C++ algorithmic differentiation*. http://www.coin-or.org/CppAD
