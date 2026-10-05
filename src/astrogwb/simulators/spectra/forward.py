"""Shared source validation and batched waveform reduction for forward models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.typing import ArrayLike
from numpy.typing import NDArray

from astrogwb.gwb.spectral import inclination_averaging_factor
from astrogwb.populations import SourceFn
from astrogwb.simulators.population.draws import bucket_size
from astrogwb.utils import array_dict_shape
from astrogwb.waveform import PolarizationPowerGenerator

#: Growth of the packed buffer's static capacity. Padding here is waveform work
#: only up to the last chunk (the loop's trip count is traced), so this ladder
#: sets how many buffer shapes -- hence compiles -- a stream of superbatches
#: sees, not how much is evaluated; it is coarser than the source-draw ladder.
PACK_RATIO = 1.5

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


class PackedPowerSum:
    """Per-draw sums of polarization power over one flat stream of sources.

    ``sources`` are flat ``(M,)`` columns holding every draw's events back to
    back and ``segment_ids`` ``(M,)`` names the draw of each. Chunks of
    ``chunk_size`` events run through the generator, and each chunk's power is
    added into a ``(num_segments, F)`` accumulator with
    :func:`jax.ops.segment_sum`, so a draw never pads to the largest one and
    only the stream's last chunk is partial. :func:`jax.lax.fori_loop`'s trip
    count is traced, so the compiled body does not depend on how many chunks a
    superbatch has; the buffer is padded on the host to a geometric capacity
    (:data:`PACK_RATIO`), so a stream of superbatches sees a few shapes, not one
    per total count. Padding rows repeat the first source -- a physical source,
    so the waveform stays finite -- and carry a segment that is dropped.

    Build once and call repeatedly: the jitted loop closes over ``generator``,
    which hashes by identity, and a fresh equal rebuild would recompile.
    Summation order follows the stream, so a draw's sum can differ at the last
    bits depending on its neighbours; its events do not.
    """

    def __init__(
        self, generator: PolarizationPowerGenerator, *, chunk_size: int
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._generator = generator
        self._chunk_size = chunk_size
        self._num_frequencies = int(np.shape(generator.frequencies)[0])
        self._reduce = jax.jit(self._loop, static_argnames="num_segments")

    def _loop(
        self,
        sources: Mapping[str, jax.Array],
        segment_ids: jax.Array,
        num_chunks: jax.Array,
        *,
        num_segments: int,
    ) -> jax.Array:
        chunk = self._chunk_size

        def body(index: jax.Array, total: jax.Array) -> jax.Array:
            start = index * chunk
            batch = {
                name: jax.lax.dynamic_slice_in_dim(values, start, chunk)
                for name, values in sources.items()
            }
            ids = jax.lax.dynamic_slice_in_dim(segment_ids, start, chunk)
            power = jnp.asarray(self._generator.generate_batch(batch))
            if power.shape != (self._num_frequencies, chunk):
                raise ValueError(
                    "waveform generator must return frequency-first power of "
                    f"shape (F, chunk) = {(self._num_frequencies, chunk)}; got "
                    f"{power.shape}"
                )
            return total + jax.ops.segment_sum(
                power.T, ids, num_segments=num_segments + 1
            )

        total = jax.lax.fori_loop(
            0,
            num_chunks,
            body,
            jnp.zeros((num_segments + 1, self._num_frequencies), dtype=jnp.float64),
        )
        return total[:num_segments]

    def __call__(
        self,
        sources: Mapping[str, ArrayLike],
        segment_ids: ArrayLike,
        num_segments: int,
    ) -> jax.Array:
        """``(num_segments, F)`` power sums; an empty stream returns zeros."""
        _require_luminosity_distance(sources)  # ty: ignore[invalid-argument-type]
        ids = np.asarray(segment_ids, dtype=np.int32)
        columns = {name: np.asarray(values) for name, values in sources.items()}
        if ids.ndim != 1 or any(v.shape != ids.shape for v in columns.values()):
            raise ValueError(
                "sources and segment_ids must be flat arrays of one length; got "
                f"{ {n: v.shape for n, v in columns.items()} } and {ids.shape}"
            )
        if ids.size and (ids.min() < 0 or ids.max() >= num_segments):
            raise ValueError(f"segment_ids must lie in [0, {num_segments})")
        total = ids.size
        if total == 0:
            return jnp.zeros((num_segments, self._num_frequencies))

        capacity = bucket_size(total, self._chunk_size, PACK_RATIO)
        pad = capacity - total
        padded = {
            name: np.concatenate([values, np.repeat(values[:1], pad)])
            for name, values in columns.items()
        }
        padded_ids = np.concatenate([ids, np.full(pad, num_segments, dtype=np.int32)])
        return self._reduce(
            padded,
            padded_ids,
            jnp.asarray(math.ceil(total / self._chunk_size)),
            num_segments=num_segments,
        )


def normalize_spectra(
    power_sums: ArrayLike,
    sources: Mapping[str, ArrayLike],
    *,
    count: Literal["poisson", "fixed"],
    total_merger_rate: NDArray[np.float64],
    observation_seconds: float,
    num_events: int | None,
) -> NDArray[np.float64]:
    r"""Turn per-draw power sums into strain spectra, ``(draws, F)``.

    ``S_h = A_{\rm inc}\, k_b \sum_{i \in b} P_i(f)`` with the per-draw factor
    ``k_b = 1/T`` for Poisson counts (the sum over the realized events divided by
    the observation time) and ``k_b = \mathcal{R}_b / N`` for fixed counts (the
    population rate times the sample mean power, so ``T`` cancels). The two
    modes share everything but this factor.
    """
    rate = np.asarray(total_merger_rate, dtype=np.float64)
    if count == "fixed":
        if num_events is None:
            raise ValueError("num_events is required for fixed counts")
        factor = rate / num_events
    else:
        factor = np.full_like(rate, 1.0 / observation_seconds)
    return (
        inclination_averaging_factor(sources)
        * factor[:, None]
        * np.asarray(power_sums, dtype=np.float64)
    )
