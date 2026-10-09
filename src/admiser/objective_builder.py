# objective_builder.py
r"""
make_builders declares the objective

    J = \int L(t, x, u, theta) dt + Phi(x(T), theta)

It only RECORDS the pieces; nothing is computed here. The solver integrates L along
the same RK4 rollout that produces every constraint integral (see ocp_to_nlp), so
the objective and the constraints are always built on one and the same
discretisation.

All constraints -- terminal, integral and path, equalities and inequalities -- are
registered through the OCPProblem.add_* API, not here.
"""

from typing import Callable, NamedTuple, Optional

from .problem_definition import DEFAULT_QUAD_SCHEME
from .quadrature import validate_quad_scheme


class ObjectiveSpec(NamedTuple):
    """
    The objective of a problem, as declared with make_builders().

    dyn  : the dynamics; must be the same function given to OCPProblem(dyn=...)
    L    : running cost L(t, x, u, theta), or None
    Phi  : terminal cost Phi(xT, theta), or None (it may also return None)
    quad : this objective's default substep quadrature scheme, see QUAD_SCHEMES
    """
    dyn: Callable
    L: Optional[Callable]
    Phi: Optional[Callable]
    quad: str


def make_builders(
    *,
    dyn,
    L=None,               # L(t, x, u, theta); may be None
    Phi=None,             # Phi(xT, theta); may be None
    # Kept only so old calls still work; no longer used to build constraints:
    terminal_eq=None,
    integral_eqs=None,
    quad: str = DEFAULT_QUAD_SCHEME,
):
    """
    Returns
    -------
    ObjectiveSpec(dyn, L, Phi, quad), to be passed as OCPProblem(objective_builder=...)

    L and Phi must be written with jax.numpy, like every model function.

    `quad` is this objective's default substep quadrature scheme; the available
    names and their orders are in quadrature.QUAD_SCHEMES. If the problem sets
    problem.quad_scheme explicitly, that wins -- the objective and every
    constraint must share one scheme, otherwise SLSQP sees them built on
    different discretisations and the resulting KKT point is meaningless.
    """
    return ObjectiveSpec(dyn=dyn, L=L, Phi=Phi, quad=validate_quad_scheme(quad))
