"""Name-to-model registries for source populations and merger-rate functions.

A catalog file records the *names* of the source and rate callables that drew
it, never an import path and never a pickled callable. Registry keys change
only on purpose; module paths change as collateral whenever a module is
moved, so a persisted ``module:function`` string is a reference that silently
rots.

Two registries, because a source model and a merger-rate function are
independent declarations: a redshift *law* used for a guard-mixture proposal
already pairs with the same Madau-Dickinson *rate* the physical population
uses. :func:`build_population` takes the source name positionally and defaults
the rate to :data:`DEFAULT_MERGER_RATE_MODEL`, so the six previously-registered
single names keep working as source keys without a recipe table.

The name pins the name, not the mathematics: re-pointing a registered key at a
different density would be invisible here. :mod:`astrogwb.catalog` carries the
drift guard that closes that hole for the redshift law.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
from jax.typing import ArrayLike

from astrogwb.populations.base import (
    MergerRateFn,
    Population,
    SourceFn,
    SourceModel,
)

__all__ = [
    "DEFAULT_MERGER_RATE_MODEL",
    "SHARED_MODEL_KWARGS",
    "build_population",
    "build_source_model",
    "known_merger_rate_models",
    "known_source_models",
    "register_merger_rate_model",
    "register_source_model",
]


_SOURCE_REGISTRY: dict[str, SourceFn] = {}
#: Registered implementations may take construction kwargs; :func:`build_population`
#: binds them into a :data:`~astrogwb.populations.base.MergerRateFn`.
_RATE_REGISTRY: dict[str, Callable[..., jax.Array]] = {}

#: Density factors a catalog selects when nothing narrower is requested.
#: Every registered source model declares ``redshift`` -- the one source
#: parameter whose density never cancels in an importance weight.
_DEFAULT_DENSITY_SITES: tuple[str, ...] = (
    "redshift",
    "source_frame_mass_1",
    "source_frame_mass_2",
)

#: The merger-rate name :func:`build_population` pairs with a source when the
#: caller names only the source. Every shipped source today uses this rate.
DEFAULT_MERGER_RATE_MODEL = "madau_dickinson"

#: Construction kwargs routed to *both* the source model and the rate
#: function: the redshift window and grid. A catalog persists one flat kwargs
#: mapping (see ``Catalog._model_kwargs``), so a name given only one of the two
#: registered callables is split by this fixed key set rather than by a
#: second persisted field. This is why ``Catalog.restrict_redshift`` can rewrite
#: ``z_min``/``z_max`` in the one flat mapping and have both halves of the
#: reconstructed population see the narrowed window.
SHARED_MODEL_KWARGS: tuple[str, ...] = ("z_min", "z_max", "n_grid")


@dataclass(frozen=True, kw_only=True)
class _BoundMergerRate:
    """Hashable ``(params) -> Array`` wrapping a registered rate plus kwargs.

    ``functools.partial`` hashes and compares by identity, so two constructions
    from the same catalog would silently retrace under ``jax.jit``. This
    wrapper is a value object: same function and kwargs, same hash.
    """

    fn: Callable[..., jax.Array]
    kwargs: tuple[tuple[str, float | int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", tuple(sorted(self.kwargs)))

    def __call__(self, params: Mapping[str, ArrayLike]) -> jax.Array:
        return self.fn(params, **dict(self.kwargs))


def register_source_model(name: str) -> Callable[[SourceFn], SourceFn]:
    """Register a source model under ``name``, returning it unchanged."""

    def decorate(fn: SourceFn) -> SourceFn:
        if name in _SOURCE_REGISTRY:
            raise ValueError(f"source model {name!r} is already registered")
        _SOURCE_REGISTRY[name] = fn
        return fn

    return decorate


def register_merger_rate_model[F: Callable[..., jax.Array]](
    name: str,
) -> Callable[[F], F]:
    """Register a merger-rate implementation under ``name``, returning it unchanged.

    The registered callable may take construction kwargs after ``params``.
    :func:`build_population` binds those into a
    :data:`~astrogwb.populations.base.MergerRateFn`.
    """

    def decorate(fn: F) -> F:
        if name in _RATE_REGISTRY:
            raise ValueError(f"merger-rate model {name!r} is already registered")
        _RATE_REGISTRY[name] = fn
        return fn

    return decorate


def build_source_model(
    name: str,
    *,
    model_kwargs: Mapping[str, float | int],
    source_kwargs: Mapping[str, float | int],
    density_sites: tuple[str, ...] = _DEFAULT_DENSITY_SITES,
) -> SourceModel:
    """Assemble the registered source model into a frozen, hashable value."""
    try:
        fn = _SOURCE_REGISTRY[name]
    except KeyError:
        known = ", ".join(known_source_models())
        raise KeyError(
            f"unknown source model {name!r}; registered models are: {known}"
        ) from None
    combined = {**dict(model_kwargs), **dict(source_kwargs)}
    return SourceModel(
        fn=fn, model_kwargs=tuple(combined.items()), density_sites=density_sites
    )


def _split_shared_kwargs(
    settings: Mapping[str, float | int],
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Split a flat kwargs mapping into the shared and source-only parts."""
    shared = {k: v for k, v in settings.items() if k in SHARED_MODEL_KWARGS}
    source_only = {k: v for k, v in settings.items() if k not in SHARED_MODEL_KWARGS}
    return shared, source_only


def _bound_merger_rate(
    name: str, *, model_kwargs: Mapping[str, float | int]
) -> MergerRateFn:
    """Look up ``name`` and bind construction kwargs as a hashable callable."""
    try:
        fn = _RATE_REGISTRY[name]
    except KeyError:
        known = ", ".join(known_merger_rate_models())
        raise KeyError(
            f"unknown merger-rate model {name!r}; registered models are: {known}"
        ) from None
    return _BoundMergerRate(fn=fn, kwargs=tuple(model_kwargs.items()))


def build_population(
    source_model: str,
    *,
    rate_model: str = DEFAULT_MERGER_RATE_MODEL,
    settings: Mapping[str, float | int] | None = None,
    source_kwargs: Mapping[str, float | int] | None = None,
    density_sites: tuple[str, ...] = _DEFAULT_DENSITY_SITES,
) -> Population:
    """Assemble a source model and a bound merger-rate function into a ``Population``.

    ``source_model`` is a registered source-model name -- the same six keys
    previously used as single population names. ``rate_model`` names any
    registered merger-rate function independently; it defaults to
    :data:`DEFAULT_MERGER_RATE_MODEL` so a caller that names only the source
    still gets the Madau-Dickinson rate every shipped source pairs with.

    ``settings`` is the flat construction-kwargs mapping a catalog persists.
    It is split automatically by :data:`SHARED_MODEL_KWARGS` between the source
    wrapper and the bound rate callable. The shared window/grid kwargs
    reach both from one definition, because a catalog persists only one flat
    kwargs mapping.
    """
    shared, source_only = _split_shared_kwargs(settings or {})
    source = build_source_model(
        source_model,
        model_kwargs=shared,
        source_kwargs={**source_only, **(source_kwargs or {})},
        density_sites=density_sites,
    )
    rate = _bound_merger_rate(rate_model, model_kwargs=shared)
    return Population(source=source, rate=rate)


def known_source_models() -> tuple[str, ...]:
    """Every registered source-model name, in sorted order."""
    return tuple(sorted(_SOURCE_REGISTRY))


def known_merger_rate_models() -> tuple[str, ...]:
    """Every registered merger-rate-model name, in sorted order."""
    return tuple(sorted(_RATE_REGISTRY))
