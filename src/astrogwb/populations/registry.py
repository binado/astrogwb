"""Name-to-population registry.

A catalog file records the *name* of the population that drew it, never an
import path and never a pickled callable. Registry keys change only on
purpose; module paths change as collateral whenever a module is moved, so a
persisted ``module:function`` string is a reference that silently rots.

One registry, because a source model and its merger rate are not independent
declarations: both are normalizations of the same redshift law, and composing
them freely is how a guard-mixture proposal came to record the plain
Madau-Dickinson rate -- a number that is not the normalization of the density
it travels with. A registered population is a *factory*: it takes the
construction settings and returns both callables at once, so the pairing is
structural rather than conventional.

A population that has no physical rate -- a guard mixture is a sampling
density, not a population -- returns ``None`` for it, and every consumer that
needs one fails by name instead of computing a meaningless scalar.

What :func:`build_population` returns is a :class:`Population` of plain
:func:`functools.partial` objects with the construction settings bound: a
:data:`SourceFn` and an optional :data:`MergerRateFn`, each called with
hyperparameters alone. Evaluating or sampling one is the job of
:func:`astrogwb.utils.sampling.evaluate_sources` and
:func:`astrogwb.utils.sampling.sample_sources`.

The factory's own signature is the settings schema: a construction key no
population takes raises ``TypeError`` here rather than being filtered away.

The name pins the name, not the mathematics: re-pointing a registered key at a
different density would be invisible here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from inspect import signature
from typing import NamedTuple

import jax
from jax.typing import ArrayLike

__all__ = [
    "DEFAULT_DENSITY_SITES",
    "MergerRateFn",
    "Population",
    "SourceFn",
    "build_population",
    "known_populations",
    "register_population",
]

#: A bound source model: declares per-source sites as a side effect and
#: returns the mapping that defines the source-output set -- the columns a
#: catalog stores. ``luminosity_distance`` is required in it: it is the
#: effective distance governing waveform amplitude.
type SourceFn = Callable[[Mapping[str, ArrayLike]], Mapping[str, jax.Array]]

#: A bound merger-rate callable: returns one observer-frame scalar, mergers per
#: second, shape ``()``, from hyperparameters alone.
type MergerRateFn = Callable[[Mapping[str, ArrayLike]], jax.Array]


class Population(NamedTuple):
    """One population's two bound callables, built together.

    ``merger_rate_fn`` is ``None`` exactly when the population declares no
    physical rate: a guard mixture fattens the tails of a proposal density and
    the Madau-Dickinson total rate is not its normalization, so there is no
    scalar to return rather than a wrong one.

    Both members hash **by identity**. Build the population once per run and
    reuse it: closing a jit-compiled function over a freshly built, equal
    callable forces a recompile.
    """

    source_model: SourceFn
    merger_rate_fn: MergerRateFn | None


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
    factory takes construction settings by keyword and returns a
    :class:`Population`; :func:`build_population` calls it.
    """

    def decorate(fn: F) -> F:
        if name in _REGISTRY:
            raise ValueError(f"population {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorate


def build_population(name: str, **settings: float) -> Population:
    """Build a registered population from its construction settings.

    ``settings`` is the flat construction mapping a catalog persists, passed
    whole: the redshift window and grid every population takes, plus whatever
    else that one takes. An unknown name raises ``KeyError`` listing the
    registered populations; a setting the named population does not take
    raises ``TypeError`` naming the population and the settings it accepts.

    The settings are bound against the factory's signature before it is
    called, so a mismatch is reported against the *population* rather than
    surfacing as a ``TypeError`` about a private factory function.

    The returned callables hash **by identity**. Build the population once per
    run and reuse it.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(known_populations())
        raise KeyError(
            f"unknown population {name!r}; registered populations are: {known}"
        ) from None
    try:
        signature(factory).bind(**settings)
    except TypeError as error:
        accepted = ", ".join(signature(factory).parameters)
        raise TypeError(
            f"population {name!r}: {error}; its construction settings are: {accepted}"
        ) from None
    return factory(**settings)


def known_populations() -> tuple[str, ...]:
    """Every registered population name, in sorted order."""
    return tuple(sorted(_REGISTRY))
