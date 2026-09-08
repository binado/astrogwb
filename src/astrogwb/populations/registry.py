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
from typing import Protocol

from jax.typing import ArrayLike

__all__ = [
    "PopulationModel",
    "PopulationModelFn",
    "known_population_models",
    "population_model",
    "register_population_model",
]


#: A registered population model, before its construction settings are bound.
#: Its first positional argument is the hyperparameter mapping; every other
#: argument is keyword-only, JSON-serializable, and static -- a redshift
#: window, a grid size. Those are what a catalog persists as its model kwargs.
#: Deliberately not a ``Protocol``: each model declares the specific
#: construction keywords it takes, which no single protocol signature can
#: describe without forcing every model to accept ``**kwargs``.
type PopulationModelFn = Callable[..., None]


class PopulationModel(Protocol):
    """A population model with its construction settings already bound.

    This is the one calling convention every consumer sees:
    ``model(params)`` executes NumPyro primitives declaring the source sites,
    the derived waveform inputs, and the population-level rate.
    """

    def __call__(self, params: Mapping[str, ArrayLike]) -> None: ...


_REGISTRY: dict[str, PopulationModelFn] = {}


def register_population_model(
    name: str,
) -> Callable[[PopulationModelFn], PopulationModelFn]:
    """Register a population model under ``name``, returning it unchanged.

    Applied directly to the model function, so the name lives beside the
    declaration rather than in a separate table that can fall out of step.
    """

    def decorate(model: PopulationModelFn) -> PopulationModelFn:
        if name in _REGISTRY:
            raise ValueError(f"population model {name!r} is already registered")
        _REGISTRY[name] = model
        return model

    return decorate


def population_model(name: str) -> PopulationModelFn:
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
