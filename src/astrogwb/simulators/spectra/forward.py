"""Shared source validation and batched waveform reduction for forward models."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.typing import ArrayLike

from astrogwb.populations import SourceFn
from astrogwb.utils import array_dict_shape
from astrogwb.waveform import PolarizationPowerGenerator

#: The source output naming the effective distance governing waveform
#: amplitude. Required in every source model's returned mapping; see
#: :mod:`astrogwb.importance.spectral`, which checks the same key.
_LUMINOSITY_DISTANCE = "luminosity_distance"


def _require_luminosity_distance(sources: Mapping[str, jax.Array]) -> None:
    """Raise unless the source mapping names the distance scaling the waveform."""
    if _LUMINOSITY_DISTANCE not in sources:
        raise KeyError(
            f"source model must return {_LUMINOSITY_DISTANCE!r}: it is the "
            "distance governing waveform amplitude"
        )


def _batch_power_sum(
    generator: PolarizationPowerGenerator,
    batch_sources: Mapping[str, jax.Array],
    event_mask: jax.Array,
) -> jax.Array:
    """Call ``generate_batch`` on one chunk and return its masked ``(F,)`` sum."""
    power = jnp.asarray(generator.generate_batch(batch_sources))
    n_chunk = next(iter(batch_sources.values())).shape[0]
    if power.ndim != 2 or power.shape[-1] != n_chunk:
        raise ValueError(
            "waveform generator must return frequency-first power of "
            f"shape (F, n_chunk); got {power.shape} for {n_chunk} sources"
        )
    return (power * event_mask).sum(axis=1)


def _sum_polarization_power(
    generator: PolarizationPowerGenerator,
    sources: Mapping[str, jax.Array],
    event_mask: ArrayLike,
    *,
    chunk_size: int,
) -> jax.Array:
    """Sum frequency-first polarization power over sources, chunk by chunk.

    Full batches of ``chunk_size`` are reduced with :func:`jax.lax.scan`, so
    the compiled body is one batch regardless of how many chunks there are; a
    static remainder is a separate ``generate_batch``. Peak waveform memory is
    ``(F, chunk_size)`` rather than ``(F, max_events)``.

    ``event_mask`` is sliced alongside each source batch, so inactive capacity
    slots do not contribute to the sum. The zero carry comes from
    ``generator.frequencies``, so an empty catalog returns zeros without
    tracing or calling a waveform kernel at all.
    """
    n_events = array_dict_shape(sources)[0]
    event_mask = jnp.asarray(event_mask)
    n_full, remainder = divmod(n_events, chunk_size)
    total = jnp.zeros(np.shape(generator.frequencies)[0], dtype=jnp.float64)

    # A Python branch, not a traced one: n_full is static, and lax.scan over
    # zero chunks would still trace the waveform body it never runs.
    if n_full:
        chunks = {
            name: values[: n_full * chunk_size].reshape(n_full, chunk_size)
            for name, values in sources.items()
        }

        chunk_masks = event_mask[: n_full * chunk_size].reshape(n_full, chunk_size)

        def accumulate(
            carry: jax.Array,
            scan_values: tuple[Mapping[str, jax.Array], jax.Array],
        ) -> tuple[jax.Array, None]:
            chunk, chunk_mask = scan_values
            return carry + _batch_power_sum(generator, chunk, chunk_mask), None

        total, _ = jax.lax.scan(
            accumulate,
            total,
            (chunks, chunk_masks),
        )
    if remainder:
        tail = {name: values[n_full * chunk_size :] for name, values in sources.items()}
        tail_mask = event_mask[n_full * chunk_size :]
        total = total + _batch_power_sum(generator, tail, tail_mask)
    return total


def validate_source_model(
    params: Mapping[str, ArrayLike],
    *,
    source_model: SourceFn,
    generator: PolarizationPowerGenerator,
    rng_key: jax.Array | int,
) -> None:
    """Check a population against a generator's waveform family, eagerly.

    The forward models cannot do this themselves: under ``jax.jit`` the
    source arrays are tracers, and a tracer cannot drive a Python exception.
    So generation trusts its inputs, and this is how a caller earns that trust
    -- one eager draw of a single source, checked against the approximant,
    before handing the model to NUTS or :class:`~numpyro.infer.Predictive`.

    No waveform is evaluated, so this is cheap enough to call unconditionally.

    Raises:
        KeyError: If ``source_model`` does not return ``luminosity_distance``.
        ValueError: If a drawn value names a degree of freedom the generator's
            approximant does not carry -- a tidal population against an
            aligned-spin model, say, whose deformabilities would otherwise be
            silently ignored.
    """
    with numpyro.handlers.seed(rng_seed=rng_key), numpyro.plate("events", 1):
        sources = dict(source_model(params))
    _require_luminosity_distance(sources)
    generator.check_sources(sources)
