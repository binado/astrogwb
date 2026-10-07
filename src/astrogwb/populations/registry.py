"""Name-to-population registry.

A catalog file records the *name* of the population that drew it, never an
import path and never a pickled callable. Registry keys change only on
purpose; module paths change as collateral whenever a module is moved, so a
persisted ``module:function`` string is a reference that silently rots.

A registered population is a *factory*: it takes the construction kwargs and
returns a :data:`~astrogwb.populations._types.Population`, a callable
``parameters -> (merger_rate, model)``. The rate and the source density come
from one call, so both are normalizations of the same redshift law by
construction. ``model()`` declares the per-source sample sites and returns the
columns a catalog stores; evaluating or sampling it is the job of
:func:`astrogwb.populations.evaluation.evaluate_sources` and
:func:`astrogwb.populations.evaluation.sample_sources`.

The factory's own signature is the kwargs schema: a construction key no
population takes raises ``TypeError`` here rather than being filtered away.

Which hyperparameters a population needs, which of them factor out of the
spectrum, and whether it is a physical population or a proposal density are
contracts documented on the factory, not checked here.

The name pins the name, not the mathematics: re-pointing a registered key at a
different density would be invisible here.
"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature

from astrogwb.populations._types import Population

__all__ = [
    "DEFAULT_DENSITY_SITES",
    "Population",
    "build_population",
    "known_populations",
    "register_population",
]

type PopulationFactory = Callable[..., Population]

_REGISTRY: dict[str, PopulationFactory] = {}

#: Density factors a catalog selects when nothing narrower is requested.
#: Every registered source model declares ``redshift`` -- the one source
#: parameter whose density never cancels in an importance weight.
DEFAULT_DENSITY_SITES: tuple[str, ...] = (
    "redshift",
    "source_frame_mass_1",
    "source_frame_mass_2",
)


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
