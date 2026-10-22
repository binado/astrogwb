"""Shared source validation and batched waveform reduction for forward models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.typing import ArrayLike
from numpy.typing import NDArray

from astrogwb.detector import DetectorSpec, stokes_cross_power
from astrogwb.populations._types import PopulationModel
from astrogwb.simulators.population.simulator import bucket_size
from astrogwb.utils import array_dict_shape
from astrogwb.waveform import PolarizationPowerGenerator

#: Growth of the padded buffer's static capacity. Padding here is waveform work
#: only up to the last chunk (the loop's trip count is traced), so this ladder
#: sets how many buffer shapes -- hence compiles -- a stream of draws sees, not
#: how much is evaluated; it is coarser than the source-draw ladder.
PACK_RATIO = 1.5

#: The source output naming the effective distance governing waveform
#: amplitude. Required in every source model's returned mapping; see
#: :mod:`astrogwb.gwb.importance`, which reads the same key.
_LUMINOSITY_DISTANCE = "luminosity_distance"

#: The extra source columns :class:`ChunkedNetworkSum` reads: the cosine of the
#: polar angle and the azimuth of the direction to the source (Earth-fixed), and
#: the polarization angle, as :func:`~astrogwb.detector.stokes_cross_power`
#: takes them.
EXTRINSIC_COLUMNS = ("polar_cosine", "azimuth", "polarization")


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
    model: PopulationModel,
    *,
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
        KeyError: If ``model`` does not return ``luminosity_distance``.
        ValueError: If a drawn value names a degree of freedom the generator's
            approximant does not carry -- a tidal population against an
            aligned-spin model, say, whose deformabilities would otherwise be
            silently ignored.
    """
    with numpyro.handlers.seed(rng_seed=rng_key), numpyro.plate("events", 1):
        sources = dict(model())
    _require_luminosity_distance(sources)
    generator.check_sources(sources)


class ChunkedPowerSum:
    """The sum of polarization power over the sources of one draw.

    ``sources`` are flat ``(n,)`` columns. Chunks of ``chunk_size`` events run
    through the generator and are added into an ``(F,)`` carry, with an event
    mask zeroing the tail of the last chunk, so only that chunk is partial.
    :func:`jax.lax.fori_loop`'s trip count is traced, so the compiled body does
    not depend on how many chunks a draw has; the buffer is padded on the host
    to a geometric capacity (:data:`PACK_RATIO`), so a stream of draws sees a
    few shapes, not one per count. Padding rows repeat the first source -- a
    physical source, so the waveform stays finite -- and are masked out.

    Build once and call repeatedly: the jitted loop closes over ``generator``,
    which hashes by identity, and a fresh equal rebuild would recompile.
    """

    def __init__(
        self, generator: PolarizationPowerGenerator, *, chunk_size: int
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._generator = generator
        self._chunk_size = chunk_size
        self._num_frequencies = int(np.shape(generator.frequencies)[0])
        self._reduce = jax.jit(self._loop)

    def _loop(self, sources: Mapping[str, jax.Array], count: jax.Array) -> jax.Array:
        chunk = self._chunk_size

        def body(index: jax.Array, total: jax.Array) -> jax.Array:
            start = index * chunk
            batch = {
                name: jax.lax.dynamic_slice_in_dim(values, start, chunk)
                for name, values in sources.items()
            }
            mask = (jnp.arange(chunk) + start) < count
            return total + _batch_power_sum(self._generator, batch, mask)

        return jax.lax.fori_loop(
            0,
            (count + chunk - 1) // chunk,
            body,
            jnp.zeros(self._num_frequencies, dtype=jnp.float64),
        )

    def __call__(self, sources: Mapping[str, ArrayLike]) -> jax.Array:
        """The ``(F,)`` power sum; an empty draw returns zeros."""
        _require_luminosity_distance(sources)  # ty: ignore[invalid-argument-type]
        padded, count = _pad_sources(sources, self._chunk_size)
        if count == 0:
            return jnp.zeros(self._num_frequencies)
        return self._reduce(padded, jnp.asarray(count))


class ChunkedNetworkSum:
    r"""The cross-power of one draw's sources in every detector pair, ``(P, F)``.

    Row :math:`p` is :math:`\sum_i K_{ab, i}(f)` for the :math:`p`-th pair
    :math:`a < b` of ``detectors``, with :math:`K_{ab, i}` the
    :func:`~astrogwb.detector.stokes_cross_power` of source :math:`i`: its
    Stokes parameters from the generator, seen through the antenna patterns
    and delay of its sky position and polarization angle. ``sources`` carry
    those as the extra columns :data:`EXTRINSIC_COLUMNS`. Chunking, padding,
    masking and compilation are those of :class:`ChunkedPowerSum`, whose
    contracts this shares; peak memory is ``(P, F, chunk_size)``.

    ``pairs`` replaces every :math:`a < b` with the index pairs given, in
    that order, so several networks can share one pass over the union of
    their detectors (:class:`~astrogwb.simulators.spectra.JointNetworkSpectralDensitySimulator`).
    """

    def __init__(
        self,
        generator: PolarizationPowerGenerator,
        detectors: Sequence[DetectorSpec],
        *,
        chunk_size: int,
        pairs: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._generator = generator
        self._detectors = list(detectors)
        self._pairs = (
            [(first, second) for first, second in pairs]
            if pairs is not None
            else [
                (first, second)
                for first in range(len(self._detectors))
                for second in range(first + 1, len(self._detectors))
            ]
        )
        self._chunk_size = chunk_size
        self._num_frequencies = int(np.shape(generator.frequencies)[0])
        self._reduce = jax.jit(self._loop)

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """The detector index pairs, in the order of the rows."""
        return list(self._pairs)

    def _loop(self, sources: Mapping[str, jax.Array], count: jax.Array) -> jax.Array:
        chunk = self._chunk_size
        frequencies = jnp.asarray(self._generator.frequencies)

        def body(index: jax.Array, total: jax.Array) -> jax.Array:
            start = index * chunk
            batch = {
                name: jax.lax.dynamic_slice_in_dim(values, start, chunk)
                for name, values in sources.items()
            }
            mask = (jnp.arange(chunk) + start) < count
            # (4, F, chunk) -> (4, chunk, F), with the padding rows zeroed.
            stokes = jnp.swapaxes(self._generator.generate_stokes_batch(batch), 1, 2)
            stokes = jnp.where(mask[None, :, None], stokes, 0.0)
            return total + jnp.stack(
                [
                    stokes_cross_power(
                        frequencies,
                        self._detectors[first],
                        self._detectors[second],
                        batch["polar_cosine"],
                        batch["azimuth"],
                        batch["polarization"],
                        stokes,
                    ).sum(axis=0)
                    for first, second in self._pairs
                ]
            )

        return jax.lax.fori_loop(
            0,
            (count + chunk - 1) // chunk,
            body,
            jnp.zeros((len(self._pairs), self._num_frequencies), dtype=jnp.float64),
        )

    def __call__(self, sources: Mapping[str, ArrayLike]) -> jax.Array:
        """The ``(P, F)`` cross-power sums; an empty draw returns zeros."""
        _require_luminosity_distance(sources)  # ty: ignore[invalid-argument-type]
        padded, count = _pad_sources(sources, self._chunk_size)
        if count == 0:
            return jnp.zeros((len(self._pairs), self._num_frequencies))
        return self._reduce(padded, jnp.asarray(count))


def _pad_sources(
    sources: Mapping[str, ArrayLike], chunk_size: int
) -> tuple[dict[str, NDArray[Any]], int]:
    """Flat columns padded to a geometric capacity, and the real source count.

    Padding rows repeat the first source -- a physical source, so the waveform
    stays finite -- and are masked out by the caller; an empty draw is returned
    unpadded.
    """
    columns = {name: np.asarray(values) for name, values in sources.items()}
    shapes = {name: values.shape for name, values in columns.items()}
    if (
        any(len(shape) != 1 for shape in shapes.values())
        or len(set(shapes.values())) != 1
    ):
        raise ValueError(f"sources must be flat arrays of one length; got {shapes}")
    n = next(iter(columns.values())).shape[0]
    if n == 0:
        return columns, 0
    pad = bucket_size(n, chunk_size, PACK_RATIO) - n
    padded = {
        name: np.concatenate([values, np.repeat(values[:1], pad)])
        for name, values in columns.items()
    }
    return padded, n


def normalize_spectra(
    power_sum: ArrayLike,
    *,
    count: Literal["poisson", "fixed"],
    total_merger_rate: float | NDArray[np.float64],
    observation_seconds: float,
    num_events: int | None,
) -> NDArray[np.float64]:
    r"""Turn one draw's power sum into a strain spectrum, ``(F,)``.

    ``S_h = k \sum_i P_i(f)`` with the factor ``k = 1/T`` for
    Poisson counts (the sum over the realized events divided by the observation
    time) and ``k = \mathcal{R} / N`` for fixed counts (the population rate
    times the sample mean power, so ``T`` cancels). The two modes share
    everything but this factor.
    """
    if count == "fixed":
        if num_events is None:
            raise ValueError("num_events is required for fixed counts")
        factor = float(total_merger_rate) / num_events
    else:
        factor = 1.0 / observation_seconds
    return factor * np.asarray(power_sum, dtype=np.float64)
