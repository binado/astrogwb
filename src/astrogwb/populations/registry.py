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

from collections.abc import Callable

from astrogwb.populations.base import Population

__all__ = [
    "PopulationFactory",
    "known_population_models",
    "population_model",
    "register_population_model",
]

# Constructors accept each population's own static construction keywords.
type PopulationFactory = Callable[..., Population]


_REGISTRY: dict[str, PopulationFactory] = {}


def register_population_model[Factory: PopulationFactory](
    name: str,
) -> Callable[[Factory], Factory]:
    """Register a population model under ``name``, returning it unchanged.

    Applied directly to the population constructor, so the name lives beside the
    declaration rather than in a separate table that can fall out of step.
    """

    def decorate(model: Factory) -> Factory:
        if name in _REGISTRY:
            raise ValueError(f"population model {name!r} is already registered")
        _REGISTRY[name] = model
        return model

    return decorate


def population_model(name: str) -> PopulationFactory:
    """Look up a registered population model, listing the known set on failure."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(known_population_models())
        raise KeyError(
            f"unknown population model {name!r}; registered models are: {known}"
        ) from None


def known_population_models() -> tuple[str, ...]:
    """Every registered population-model name, in sorted order."""
    return tuple(sorted(_REGISTRY))
