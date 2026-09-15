"""Name-to-model registries for source models and merger-rate functions.

A catalog file records the *names* of the source and rate callables that drew
it, never an import path and never a pickled callable. Registry keys change
only on purpose; module paths change as collateral whenever a module is
moved, so a persisted ``module:function`` string is a reference that silently
rots.

Two registries, because a source model and a merger-rate function are
independent declarations: a redshift *law* used for a guard-mixture proposal
already pairs with the same Madau-Dickinson *rate* the physical population
uses. One source registry holds both physical and proposal models -- a
proposal is just the source model a catalog happened to be drawn from.

What the builders return is a plain :func:`functools.partial` with the
construction settings bound: a :data:`SourceFn` or a :data:`MergerRateFn`,
called with hyperparameters alone. Evaluating or sampling one is the job of
:func:`astrogwb.utils.sampling.evaluate_sources` and
:func:`astrogwb.utils.sampling.sample_sources`.

The name pins the name, not the mathematics: re-pointing a registered key at a
different density would be invisible here. :mod:`astrogwb.catalog` carries the
drift guard that closes that hole for the redshift law.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import jax
from jax.typing import ArrayLike

__all__ = [
    "DEFAULT_DENSITY_SITES",
    "DEFAULT_MERGER_RATE_MODEL",
    "REDSHIFT_SITE",
    "SHARED_MODEL_KWARGS",
    "MergerRateFn",
    "SourceFn",
    "build_merger_rate_fn",
    "build_source_model",
    "known_merger_rate_models",
    "known_source_models",
    "register_merger_rate_model",
    "register_source_model",
]

#: A bound source model: declares per-source sites as a side effect and
#: returns the mapping that defines the source-output set -- the columns a
#: catalog stores. ``luminosity_distance`` is required in it: it is the
#: effective distance governing waveform amplitude.
type SourceFn = Callable[[Mapping[str, ArrayLike]], Mapping[str, jax.Array]]

#: A bound merger-rate callable: returns one observer-frame scalar, mergers per
#: second, shape ``()``, from hyperparameters alone.
type MergerRateFn = Callable[[Mapping[str, ArrayLike]], jax.Array]

_SOURCE_REGISTRY: dict[str, Callable[..., Mapping[str, jax.Array]]] = {}
_RATE_REGISTRY: dict[str, Callable[..., jax.Array]] = {}

#: The redshift site every source model must declare. Its density can never be
#: excluded: redshift is the one source parameter the target and the proposal
#: are guaranteed to disagree on. It lives beside the registry because both the
#: populations that declare it and the catalogs that store it need the name, and
#: :mod:`astrogwb.catalog` imports this package rather than the other way round;
#: it is re-exported as ``astrogwb.catalog.REDSHIFT_SITE``.
REDSHIFT_SITE = "redshift"

#: Density factors a catalog selects when nothing narrower is requested.
#: Every registered source model declares :data:`REDSHIFT_SITE` -- the one
#: source parameter whose density never cancels in an importance weight.
DEFAULT_DENSITY_SITES: tuple[str, ...] = (
    REDSHIFT_SITE,
    "source_frame_mass_1",
    "source_frame_mass_2",
)

#: The merger-rate name :func:`build_merger_rate_fn` resolves when the caller
#: names none. Every shipped source model pairs with this rate.
DEFAULT_MERGER_RATE_MODEL = "madau_dickinson"

#: Construction kwargs routed to *both* the source model and the rate
#: function: the redshift window and grid. A catalog persists one flat kwargs
#: mapping (see ``PopulationRecord.model_kwargs``), so a name given only one of
#: the two registered callables is split by this fixed key set rather than by
#: a second persisted field. This is why
#: ``PolarizationPowerCatalog.restrict_redshift`` can rewrite
#: ``z_min``/``z_max`` in the one flat mapping and have both reconstructed
#: callables see the narrowed window.
SHARED_MODEL_KWARGS: tuple[str, ...] = ("z_min", "z_max", "n_grid")


def register_source_model[F: Callable[..., Mapping[str, jax.Array]]](
    name: str,
) -> Callable[[F], F]:
    """Register a source model under ``name``, returning it unchanged.

    Physical and proposal models share this one registry. The registered
    callable takes ``params`` positionally and construction kwargs by keyword;
    :func:`build_source_model` binds the latter.
    """

    def decorate(fn: F) -> F:
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
    :func:`build_merger_rate_fn` binds those into a :data:`MergerRateFn`.
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
    settings: Mapping[str, float | int] | None = None,
    source_kwargs: Mapping[str, float | int] | None = None,
) -> SourceFn:
    """Bind a registered source model's construction settings.

    ``settings`` is the flat construction-kwargs mapping a catalog persists,
    shared window/grid keys and source-only keys alike; ``source_kwargs`` adds
    source-only keys on top, overriding ``settings`` on a shared key -- the
    same merge a generated catalog persists. An unknown name raises
    ``KeyError`` listing the registered models.

    The returned :func:`functools.partial` is compared and hashed **by
    identity**. Build it once per run and reuse it: closing a jit-compiled
    function over a freshly built, equal model forces a recompile.
    """
    try:
        fn = _SOURCE_REGISTRY[name]
    except KeyError:
        known = ", ".join(known_source_models())
        raise KeyError(
            f"unknown source model {name!r}; registered models are: {known}"
        ) from None
    return partial(fn, **{**(settings or {}), **(source_kwargs or {})})


def build_merger_rate_fn(
    name: str = DEFAULT_MERGER_RATE_MODEL,
    *,
    settings: Mapping[str, float | int] | None = None,
) -> MergerRateFn:
    """Bind a registered merger-rate function's construction settings.

    Only the :data:`SHARED_MODEL_KWARGS` keys of ``settings`` are bound, so the
    same flat mapping that builds a source model builds its rate. An unknown
    name raises ``KeyError`` listing the registered rates.

    Like :func:`build_source_model`, the returned partial hashes by identity:
    build it once per run.
    """
    try:
        fn = _RATE_REGISTRY[name]
    except KeyError:
        known = ", ".join(known_merger_rate_models())
        raise KeyError(
            f"unknown merger-rate model {name!r}; registered models are: {known}"
        ) from None
    shared = {k: v for k, v in (settings or {}).items() if k in SHARED_MODEL_KWARGS}
    return partial(fn, **shared)


def known_source_models() -> tuple[str, ...]:
    """Every registered source-model name, in sorted order."""
    return tuple(sorted(_SOURCE_REGISTRY))


def known_merger_rate_models() -> tuple[str, ...]:
    """Every registered merger-rate-model name, in sorted order."""
    return tuple(sorted(_RATE_REGISTRY))
