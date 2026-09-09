"""Name-to-model registry for source populations.

A catalog file records the *name* of the population model that drew it, never
an import path and never a pickled callable. Registry keys change only on
purpose; module paths change as collateral whenever a module is moved, so a
persisted ``module:function`` string is a reference that silently rots.

The name pins the name, not the mathematics: re-pointing a registered key at a
different density would be invisible here. :mod:`astrogwb.catalog` carries the
drift guard that closes that hole for the redshift law.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from astrogwb.populations.base import Population, PopulationFn

__all__ = [
    "build_population",
    "known_population_models",
    "register_population_model",
]


_REGISTRY: dict[str, tuple[PopulationFn, tuple[str, ...]]] = {}

#: Density factors a catalog selects when nothing narrower is requested.
#: Every registered population declares ``redshift`` -- the one source
#: parameter whose density never cancels in an importance weight.
_DEFAULT_DENSITY_SITES: tuple[str, ...] = ("redshift",)


def register_population_model(
    name: str, *, source_sites: tuple[str, ...]
) -> Callable[[PopulationFn], PopulationFn]:
    """Register a population model under ``name``, returning it unchanged.

    Applied directly to the model function, so the name and its declared
    source outputs live beside the declaration rather than in a separate table
    that can fall out of step.
    """

    def decorate(fn: PopulationFn) -> PopulationFn:
        if name in _REGISTRY:
            raise ValueError(f"population model {name!r} is already registered")
        _REGISTRY[name] = (fn, tuple(source_sites))
        return fn

    return decorate


def build_population(
    name: str,
    *,
    settings: Mapping[str, float | int],
    density_sites: tuple[str, ...] = _DEFAULT_DENSITY_SITES,
) -> Population:
    """Assemble the registered model into a frozen, hashable ``Population``.

    ``source_sites`` is not a caller-supplied argument: which sites a model
    declares is a property of the model, stored on the registry entry, not an
    analysis choice like ``density_sites``.
    """
    try:
        fn, source_sites = _REGISTRY[name]
    except KeyError:
        known = ", ".join(known_population_models())
        raise KeyError(
            f"unknown population model {name!r}; registered models are: {known}"
        ) from None
    return Population(
        fn=fn,
        settings=tuple(settings.items()),
        density_sites=density_sites,
        source_sites=source_sites,
    )


def known_population_models() -> tuple[str, ...]:
    """Every registered population-model name, in sorted order."""
    return tuple(sorted(_REGISTRY))
