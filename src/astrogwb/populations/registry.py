"""Name-to-model registries for source populations and merger-rate models.

A catalog file records the *names* of the source and rate models that drew
it, never an import path and never a pickled callable. Registry keys change
only on purpose; module paths change as collateral whenever a module is
moved, so a persisted ``module:function`` string is a reference that silently
rots.

Two registries, because a source model and a merger-rate model are
independent declarations: a redshift *law* used for a guard-mixture proposal
already pairs with the same Madau-Dickinson *rate* the physical population
uses. ``_RECIPES`` keeps the six previously-registered single names working
as a fixed pairing, so catalog reconstruction, the notebooks, and every
existing call site naming one of them keep working unchanged.

The name pins the name, not the mathematics: re-pointing a registered key at a
different density would be invisible here. :mod:`astrogwb.catalog` carries the
drift guard that closes that hole for the redshift law.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from astrogwb.populations.base import (
    MergerRateFn,
    MergerRateModel,
    Population,
    SourceFn,
    SourceModel,
)

__all__ = [
    "SHARED_MODEL_KWARGS",
    "build_merger_rate_model",
    "build_population",
    "build_source_model",
    "known_merger_rate_models",
    "known_source_models",
    "recipe_name",
    "register_merger_rate_model",
    "register_recipe",
    "register_source_model",
    "resolve_recipe",
]


_SOURCE_REGISTRY: dict[str, SourceFn] = {}
_RATE_REGISTRY: dict[str, MergerRateFn] = {}
#: Legacy single population name -> (source model name, rate model name).
_RECIPES: dict[str, tuple[str, str]] = {}

#: Density factors a catalog selects when nothing narrower is requested.
#: Every registered source model declares ``redshift`` -- the one source
#: parameter whose density never cancels in an importance weight.
_DEFAULT_DENSITY_SITES: tuple[str, ...] = (
    "redshift",
    "source_frame_mass_1",
    "source_frame_mass_2",
)

#: Construction kwargs routed to *both* the source and the rate model: the
#: redshift window and grid. A catalog persists one flat kwargs mapping (see
#: ``Catalog._model_kwargs``), so a name given only one of the two registered
#: models -- or a legacy recipe name -- is split by this fixed key set rather
#: than by a second persisted field. This is why ``Catalog.restrict_redshift``
#: can rewrite ``z_min``/``z_max`` in the one flat mapping and have both halves
#: of the reconstructed population see the narrowed window.
SHARED_MODEL_KWARGS: tuple[str, ...] = ("z_min", "z_max", "n_grid")


def register_source_model(name: str) -> Callable[[SourceFn], SourceFn]:
    """Register a source model under ``name``, returning it unchanged."""

    def decorate(fn: SourceFn) -> SourceFn:
        if name in _SOURCE_REGISTRY:
            raise ValueError(f"source model {name!r} is already registered")
        _SOURCE_REGISTRY[name] = fn
        return fn

    return decorate


def register_merger_rate_model(name: str) -> Callable[[MergerRateFn], MergerRateFn]:
    """Register a merger-rate model under ``name``, returning it unchanged."""

    def decorate(fn: MergerRateFn) -> MergerRateFn:
        if name in _RATE_REGISTRY:
            raise ValueError(f"merger-rate model {name!r} is already registered")
        _RATE_REGISTRY[name] = fn
        return fn

    return decorate


def register_recipe(name: str, *, source_model: str, rate_model: str) -> None:
    """Register ``name`` as a fixed (source, rate) pairing.

    Does not validate that ``source_model`` / ``rate_model`` are themselves
    registered: recipes are declared at import time, in the same module that
    registers the source models, before the rate module they pair with is
    necessarily imported.
    """
    if name in _RECIPES:
        raise ValueError(f"population recipe {name!r} is already registered")
    _RECIPES[name] = (source_model, rate_model)


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


def build_merger_rate_model(
    name: str, *, model_kwargs: Mapping[str, float | int]
) -> MergerRateModel:
    """Assemble the registered merger-rate model into a frozen, hashable value."""
    try:
        fn = _RATE_REGISTRY[name]
    except KeyError:
        known = ", ".join(known_merger_rate_models())
        raise KeyError(
            f"unknown merger-rate model {name!r}; registered models are: {known}"
        ) from None
    return MergerRateModel(fn=fn, model_kwargs=tuple(dict(model_kwargs).items()))


def _split_shared_kwargs(
    settings: Mapping[str, float | int],
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Split a flat kwargs mapping into the shared and source-only parts."""
    shared = {k: v for k, v in settings.items() if k in SHARED_MODEL_KWARGS}
    source_only = {k: v for k, v in settings.items() if k not in SHARED_MODEL_KWARGS}
    return shared, source_only


def build_population(
    name: str | None = None,
    *,
    source_model: str | None = None,
    rate_model: str | None = None,
    settings: Mapping[str, float | int] | None = None,
    source_kwargs: Mapping[str, float | int] | None = None,
    density_sites: tuple[str, ...] = _DEFAULT_DENSITY_SITES,
) -> Population:
    """Assemble a source model and a rate model into a ``Population``.

    Two call shapes:

    - Legacy/recipe: ``name`` is one of the registered recipe names (the six
      previously-registered population names), resolving to a fixed
      (source, rate) pair. ``settings`` is the flat construction-kwargs
      mapping a catalog persists (and every existing call site already
      passes); it is split automatically by :data:`SHARED_MODEL_KWARGS`
      between the two models.
    - Explicit: ``source_model`` and ``rate_model`` name any registered pair
      independently -- the composition a run config expresses when it names
      the two roles separately, rather than one recipe.

    Either way the shared window/grid kwargs reach both models from one
    definition, because a catalog persists only one flat kwargs mapping.
    """
    if name is not None:
        if source_model is not None or rate_model is not None:
            raise ValueError(
                "build_population takes either name or source_model/rate_model, "
                "not both"
            )
        source_model, rate_model = resolve_recipe(name)
    elif source_model is None or rate_model is None:
        raise ValueError(
            "build_population requires both source_model and rate_model when "
            "name is not given"
        )

    shared, source_only = _split_shared_kwargs(settings or {})
    source = build_source_model(
        source_model,
        model_kwargs=shared,
        source_kwargs={**source_only, **(source_kwargs or {})},
        density_sites=density_sites,
    )
    rate = build_merger_rate_model(rate_model, model_kwargs=shared)
    return Population(source=source, rate=rate)


def known_source_models() -> tuple[str, ...]:
    """Every registered source-model name, in sorted order."""
    return tuple(sorted(_SOURCE_REGISTRY))


def known_merger_rate_models() -> tuple[str, ...]:
    """Every registered merger-rate-model name, in sorted order."""
    return tuple(sorted(_RATE_REGISTRY))


def resolve_recipe(name: str) -> tuple[str, str]:
    """The ``(source_model, rate_model)`` pair a recipe name resolves to.

    Shared by :func:`build_population`'s legacy call shape and by
    :mod:`astrogwb.catalog._io`'s read path, which maps an older catalog's
    single ``population_model`` attribute through the same table.
    """
    try:
        return _RECIPES[name]
    except KeyError:
        known = ", ".join(sorted(_RECIPES))
        raise KeyError(
            f"unknown population model {name!r}; registered models are: {known}"
        ) from None


def recipe_name(source_model: str, rate_model: str) -> str | None:
    """The registered recipe name pairing ``(source_model, rate_model)``, if any.

    A reverse lookup over :func:`register_recipe`'s table, used to recover a
    single legacy name for logging and for a catalog whose (source, rate)
    pair happens to match one of the six shipped recipes. Returns ``None``
    for a pairing no recipe names -- an arbitrary composition a run config
    can express that has no single legacy name.
    """
    for name, pair in _RECIPES.items():
        if pair == (source_model, rate_model):
            return name
    return None
