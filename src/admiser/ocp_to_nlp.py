# ocp_to_nlp.py
"""
Transcribe an optimal control problem into the functions an NLP solver needs.

What is built
-------------
One function of the decision vector z = [U (N*nu) ; theta (ntheta) ; tau (N)],

    F(z) = [ J(z) ; G(z) ; C(z) ]

with J the objective, G the equality constraints (which must be zero) and C the
inequality constraints (which must be non-negative), plus its Jacobian dF/dz.

Everything comes from ONE simulation of the system. The objective integral and
every constraint integral are accumulated along the same RK4 rollout, so the
objective and the constraints can never end up on different discretisations --
a mismatch that would not raise, but would produce a meaningless KKT point.

How the derivatives are obtained
--------------------------------
F is an ordinary Python function written with jax.numpy, and so are the user's
model functions inside it. JAX can transform such a function: jax.jit compiles it
into fast machine code, and jax.vjp differentiates it in REVERSE mode.

Reverse mode is the right choice for control parametrization. Its cost grows with
the number of OUTPUTS -- one objective plus a handful of constraints -- and hardly
at all with the number of inputs, which here are the N*nu control values and
easily number in the hundreds. Forward mode is the other way round: one pass per
input.

One flat loop over the substeps
-------------------------------
The rollout is a single jax.lax.scan over all N*m_sub RK4 substeps. scan compiles
the loop body ONCE and then repeats it, so the compile time does not grow with N.
jax.checkpoint on the body makes the reverse pass keep only the state and the
running integrals of each substep, recomputing everything else on the way back,
instead of storing every intermediate value.

Both choices were measured against the obvious alternative -- a loop over the
segments that unrolls each segment's m_sub substeps -- which took several seconds
(up to 16 s) to compile and differentiated 5-34x slower than the CppAD tapes this
module replaces. In the form below compiling takes under a second, and the
Jacobian is about as fast as CppAD's, faster on the larger examples.

eps and gamma are arguments, not constants
------------------------------------------
The smoothing width eps and the tolerance gamma of every path inequality are
passed into F when it is called. jax.jit recompiles only when the SHAPE of an
argument changes, never when its value does, so all the rounds of an eps -> 0
continuation share one compilation.
"""

import inspect
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .constraint_smoothing import L_eps, smooth_abs
from .quadrature import rk4_substep


class CompiledNLP(NamedTuple):
    """
    The transcribed NLP: compiled functions plus the layout of their output.

    values(z, eps, gamma)              -> F(z) = [J ; G ; C]
    values_and_jacobian(z, eps, gamma) -> (F(z), dF/dz)
    trajectory(z)                      -> (t, X): the time grid, shape (N+1,), and
                                          the state at every segment boundary,
                                          shape (N+1, nx)
    n_eq, n_ineq                       -> number of rows in G and in C

    eps and gamma hold one value per path inequality, in registration order.
    """
    values: Callable
    values_and_jacobian: Callable
    trajectory: Callable
    n_eq: int
    n_ineq: int

    @property
    def obj_slice(self):
        """Rows holding the objective (always exactly one row)."""
        return slice(0, 1)

    @property
    def eq_slice(self):
        """Rows holding G(z)."""
        return slice(1, 1 + self.n_eq)

    @property
    def ineq_slice(self):
        """Rows holding C(z)."""
        return slice(1 + self.n_eq, 1 + self.n_eq + self.n_ineq)


#: Shown when a model function cannot be traced. These errors come from inside the
#: user's own code, and JAX's message alone does not say what to change there.
_JAX_HINT = (
    "A model function (dyn, L, Phi, a constraint function or x0_from_theta) could "
    "not be traced by JAX. Model functions must be written with jax.numpy:\n"
    "  - build vectors with jnp.array([...]), not np.array([...], dtype=object)\n"
    "  - use jnp.exp, jnp.sin, jnp.sqrt, ... instead of the np.* versions\n"
    "  - replace an if/else that tests a value with jnp.where(condition, a, b)\n"
    "  - do not call float() or int() on a state, a control or a parameter\n"
    "See 'Writing Model Functions for JAX' in the README."
)

#: The JAX errors that mean "a model function did something a tracer cannot do".
#: Looked up by name because not every JAX version defines all of them.
_TRACING_ERRORS = tuple(
    getattr(jax.errors, name)
    for name in ("ConcretizationTypeError", "TracerArrayConversionError",
                 "TracerBoolConversionError", "TracerIntegerConversionError")
    if hasattr(jax.errors, name))


def _with_theta(dyn):
    """
    The dynamics may be written as dyn(x, u) or as dyn(x, u, theta). Return a
    version that always takes the three arguments.
    """
    if len(inspect.signature(dyn).parameters) == 2:
        return lambda x, u, theta: dyn(x, u)
    return dyn


def _scalar(value, what):
    """Turn what a model function returned into one number, insisting it is one."""
    value = jnp.asarray(value, dtype=float)
    if value.size != 1:
        raise ValueError(f"{what} must return a single number, "
                         f"but it returned an array of shape {value.shape}")
    return value.reshape(())


def _vector(value):
    """Turn what a terminal function returned into a 1-D array."""
    return jnp.atleast_1d(jnp.asarray(value, dtype=float))


def build_nlp(problem) -> CompiledNLP:
    """
    Transcribe `problem` and return its compiled NLP functions.

    Nothing is evaluated here. The functions are traced once, to check the model
    functions and to learn the size of G and C; JAX compiles them on their first
    call.
    """
    problem.validate()

    N, nu, nx, m_sub = problem.N, problem.nu, problem.nx, problem.m_sub
    n_theta = problem.ntheta if problem.has_params() else 0
    n_z = N * nu + n_theta + problem.n_tau
    n_path = len(problem.path_ineq_specs)

    # The objective and every constraint must share one quadrature scheme.
    # Resolving it once, here, is what guarantees that.
    scheme = problem.resolve_quad_scheme()
    L = problem.objective_builder.L
    Phi = problem.objective_builder.Phi
    dyn = _with_theta(problem.dyn)

    # ------------------------------------------------------------------
    # Every integral the problem needs, accumulated side by side in ONE vector,
    # in this order:
    #
    #   [ int L dt | integral eqs | path eqs | integral ineqs | path ineqs ]
    #
    # parts() below reads them back in the same order.
    # ------------------------------------------------------------------
    n_int = ((L is not None) + len(problem.int_eq_specs) + len(problem.path_eq_specs)
             + len(problem.int_ineq_specs) + n_path)

    def integrands(t, x, u, theta, eps):
        """Every integrand of the problem at one quadrature sample point, as a vector."""
        values = []
        if L is not None:
            values.append(_scalar(L(t, x, u, theta), "L"))
        for spec in problem.int_eq_specs:
            values.append(_scalar(spec["qfun"](t, x, u, theta), "an add_integral_eq qfun"))
        for spec in problem.path_eq_specs:
            h = _scalar(spec["hfun"](t, x, u, theta), "an add_path_eq hfun")
            values.append(h * h if spec["mode"] == "L2" else smooth_abs(h, spec["eps_abs"]))
        for spec in problem.int_ineq_specs:
            values.append(_scalar(spec["qfun"](t, x, u, theta), "an add_integral_ineq qfun"))
        for j, spec in enumerate(problem.path_ineq_specs):
            h = _scalar(spec["hfun"](t, x, u, theta), "an add_path_ineq hfun")
            # eps[j] is this round's smoothing width, passed in at call time.
            values.append(L_eps(h, eps[j]))
        return jnp.stack(values)

    def split(z):
        """Cut z into the controls (one row per segment), theta and the durations."""
        U = z[:N * nu].reshape(N, nu)
        theta = z[N * nu: N * nu + n_theta] if n_theta else None
        if problem.has_time_scaling():
            # CPET: the durations are decision variables. The substep lengths, the
            # quadrature weights and the clock below all become functions of them,
            # which is exactly the transformed system dx/ds = tau_k*f, dt/ds = tau_k.
            durations = z[N * nu + n_theta:]
        else:
            durations = jnp.full(N, problem.dt)
        return U, theta, durations

    def simulate(z, eps, record_states):
        """
        The single RK4 rollout behind everything.

        Returns (x0, xT, integrals, theta, durations, states). `integrals` is the
        vector described above; it is left empty when record_states is set, since
        replaying the trajectory needs no integrals. `states` holds the state after
        every substep when record_states is set, and is None otherwise.
        """
        U, theta, durations = split(z)

        x0 = problem.initial_state(theta)
        if x0.shape != (nx,):
            raise ValueError(f"the initial state must have shape ({nx},), got {x0.shape}")

        def f(x, u):
            dx = jnp.asarray(dyn(x, u, theta), dtype=float)
            if dx.shape != (nx,):
                raise ValueError(f"dyn must return an array of shape ({nx},), "
                                 f"but it returned shape {dx.shape}")
            return dx

        # The scan walks over one row per substep: its start time, its length and
        # the control held on it (constant across the m_sub substeps of a segment).
        h = durations / m_sub
        segment_start = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durations)[:-1]])
        t_sub = (segment_start[:, None] + h[:, None] * jnp.arange(m_sub)).reshape(-1)
        h_sub = jnp.repeat(h, m_sub)
        u_sub = jnp.repeat(U, m_sub, axis=0)

        integrate = n_int > 0 and not record_states

        def substep(carry, row):
            """One RK4 substep, plus its share of every integral."""
            x, integrals = carry
            t, h_k, u = row
            x_end, samples = rk4_substep(x, u, h_k, f, t=t, quad=scheme)
            if integrate:
                # The weights w_i of a substep add up to its length, so summing
                # w_i * integrand over all substeps yields the integrals.
                for t_i, x_i, w_i in samples:
                    integrals = integrals + w_i * integrands(t_i, x_i, u, theta, eps)
            return (x_end, integrals), (x_end if record_states else None)

        start = (x0, jnp.zeros(n_int if integrate else 0))
        (xT, integrals), states = jax.lax.scan(jax.checkpoint(substep), start,
                                               (t_sub, h_sub, u_sub))
        return x0, xT, integrals, theta, durations, states

    def parts(z, eps, gamma):
        """J, G and C, all from one rollout."""
        _, xT, acc, theta, _, _ = simulate(z, eps, record_states=False)
        i = 0  # read position in acc

        # ---------------- objective:  int L dt + Phi(xT) ----------------
        J = jnp.asarray(0.0)
        if L is not None:
            J = J + acc[i]
            i += 1
        if Phi is not None:
            phi = Phi(xT, theta)
            if phi is not None:   # the template spells "no terminal cost" as returning None
                J = J + _scalar(phi, "Phi")

        # ---------------- equalities G(z) = 0 ----------------
        # The order below is the order eq_resid is reported in, so it is part of
        # the observable behaviour and must not be shuffled.
        G = [_vector(spec["psi"](xT, theta)) for spec in problem.term_eq_specs]
        for spec in problem.int_eq_specs:              # int q dt = target
            G.append(_vector(acc[i] - spec["target"]))
            i += 1
        for _spec in problem.path_eq_specs:            # int h^2 dt (or smooth|h|) = 0
            G.append(_vector(acc[i]))
            i += 1

        # ---------------- inequalities C(z) >= 0 ----------------
        C = []
        for spec in problem.term_ineq_specs:           # phi <= 0 becomes -phi >= 0
            phi = _vector(spec["phi"](xT, theta))
            C.append(-phi if spec["sense"] == "<=" else phi)
        for spec in problem.int_ineq_specs:            # "<=": bound - int q >= 0
            q = acc[i]
            i += 1
            C.append(_vector(spec["bound"] - q if spec["sense"] == "<=" else q - spec["bound"]))
        for j in range(n_path):                        # gamma - int L_eps(h) dt >= 0
            C.append(_vector(gamma[j] - acc[i]))
            i += 1

        empty = jnp.zeros(0)
        return (J,
                jnp.concatenate(G) if G else empty,
                jnp.concatenate(C) if C else empty)

    def values(z, eps, gamma):
        """F(z) = [J ; G ; C] as one vector."""
        J, G, C = parts(z, eps, gamma)
        return jnp.concatenate([J[None], G, C])

    def values_and_jacobian(z, eps, gamma):
        """F(z) and its Jacobian, by reverse-mode AD."""
        out, pullback = jax.vjp(lambda zz: values(zz, eps, gamma), z)
        # Reverse mode: feeding the backward pass row i of the identity returns the
        # gradient of output i. jax.vmap runs all those rows together as one batched
        # backward pass, so a single sweep yields the whole Jacobian.
        (jacobian,) = jax.vmap(pullback)(jnp.eye(out.shape[0], dtype=out.dtype))
        return out, jacobian

    def trajectory(z):
        """The time grid and the state at every segment boundary, for reporting."""
        x0, _, _, _, durations, states = simulate(z, jnp.zeros(n_path), record_states=True)
        X = jnp.concatenate([x0[None, :], states[m_sub - 1::m_sub]])
        t = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durations)])
        return t, X

    # Trace once -- without compiling or running anything -- to learn the size of
    # G and C (a terminal function may return a vector), and to report a mistake in
    # a model function now, with a hint, instead of from deep inside SciPy later.
    z_spec = jax.ShapeDtypeStruct((n_z,), jnp.float64)
    p_spec = jax.ShapeDtypeStruct((n_path,), jnp.float64)
    try:
        _, G_spec, C_spec = jax.eval_shape(parts, z_spec, p_spec, p_spec)
    except _TRACING_ERRORS as exc:
        raise TypeError(f"{_JAX_HINT}\n\nJAX reported: {exc}") from exc

    return CompiledNLP(values=jax.jit(values),
                       values_and_jacobian=jax.jit(values_and_jacobian),
                       trajectory=jax.jit(trajectory),
                       n_eq=G_spec.shape[0], n_ineq=C_spec.shape[0])
