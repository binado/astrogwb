"""Name-to-population registry.

A registered population is a *factory*: it takes the construction kwargs and
returns a :data:`~astrogwb.populations._types.Population`, a callable
``parameters -> (redshift_distribution, source_model)``. The merger rate is
``redshift_distribution.total_merger_rate()``, so rate and redshift density are
normalizations of the same law by construction. ``source_model()`` declares the
intrinsic sample sites and knows nothing about redshift;
:func:`~astrogwb.populations.joint.joint_model` composes both halves into the
model a catalog is drawn from. Evaluating or sampling that is the job of
:func:`astrogwb.populations.evaluation.evaluate_sources` and
:func:`astrogwb.populations.evaluation.sample_sources`.

The factory's own signature is the kwargs schema: a construction key no
population takes raises ``TypeError`` here rather than being filtered away.

Which hyperparameters a population needs, which of them factor out of the
spectrum, and whether it is a physical population or a proposal density are
contracts documented on the factory, not checked here.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature

from astrogwb.populations._types import Population

__all__ = [
    "DEFAULT_DENSITY_SITES",
    "INTRINSIC_DENSITY_SITES",
    "Population",
    "build_population",
    "known_populations",
    "register_population",
]

type PopulationFactory = Callable[..., Population]

_REGISTRY: dict[str, PopulationFactory] = {}

#: The intrinsic density factors of a source model: the ordered source-frame
#: mass pair. These are the factors an importance weight divides by, since
#: redshift is integrated by quadrature rather than reweighted.
INTRINSIC_DENSITY_SITES: tuple[str, ...] = (
    "source_frame_mass_1",
    "source_frame_mass_2",
)

#: Density factors of the joint model a catalog selects when nothing narrower
#: is requested: ``redshift`` plus the intrinsic ones.
DEFAULT_DENSITY_SITES: tuple[str, ...] = ("redshift", *INTRINSIC_DENSITY_SITES)


def register_population[F: PopulationFactory](name: str) -> Callable[[F], F]:
    """Register a population factory under ``name``, returning it unchanged.

    Physical and proposal populations share this one registry -- a proposal is
    just the population a catalog happened to be drawn from. The registered
    factory takes construction kwargs by keyword and returns a
    :data:`Population`; :func:`build_population` calls it.
    """

    def decorate(fn: F) -> F:
        if name in _REGISTRY:
            raise ValueError(f"population {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorate


def build_population(name: str, **kwargs: float | bool | str) -> Population:
    """Build a registered population from its construction kwargs.

    ``kwargs`` is the flat construction mapping a catalog persists, passed
    whole: the redshift window and grid every population takes, plus whatever
    else that one takes. An unknown name raises ``KeyError`` listing the
    registered populations; a kwarg the named population does not take
    raises ``TypeError`` naming the population and the kwargs it accepts.

    The kwargs are bound against the factory's signature before it is
    called, so a mismatch is reported against the *population* rather than
    surfacing as a ``TypeError`` about a private factory function.

    The returned callable hashes **by identity**. Build the population once per
    run and reuse it: closing a jit-compiled function over a freshly built, equal
    callable forces a recompile.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(known_populations())
        raise KeyError(
            f"unknown population {name!r}; registered populations are: {known}"
        ) from None
    try:
        signature(factory).bind(**kwargs)
    except TypeError as error:
        accepted = ", ".join(signature(factory).parameters)
        raise TypeError(
            f"population {name!r}: {error}; its construction kwargs are: {accepted}"
        ) from None
    return factory(**kwargs)


def known_populations() -> tuple[str, ...]:
    """Every registered population name, in sorted order."""
    return tuple(sorted(_REGISTRY))
