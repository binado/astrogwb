r"""The background stage: per-pair cross-power spectra of one realization.

Source :math:`i` has radiation-frame polarizations
:math:`(\tilde h_{+,i}, \tilde h_{\times,i})` determined by its intrinsic
parameters, inclination and coalescence phase (strain in :math:`\mathrm{Hz}^{-1}`,
Fourier convention :math:`\tilde h(f) = \int h(t)\, e^{-2\pi i f t}\, dt`). It
reaches detector :math:`a` as
:math:`H_{a,i} = F^+_{a,i} \tilde h_{+,i} + F^\times_{a,i} \tilde h_{\times,i}`,
with antenna patterns of its sky position and polarization angle, and leaves
in the pair :math:`(a, b)` the real cross-power

.. math::

    K_{ab,i} = \mathrm{Re}\left[e^{i\phi_{ab,i}} H_{a,i}^* H_{b,i}\right],

:math:`\phi_{ab,i}` the inter-site delay phase. The deterministic projection
sums the individual powers, :math:`D_{ab} = \sum_i K_{ab,i}`, and normalizes
by the observation time :math:`T` as

.. math::

    Y_{ab} = \frac{5\, D_{ab}}{T}.

Over an isotropic sky and uniform polarization angle
:math:`\langle K_{ab} \rangle = I \gamma_{ab} / 5`, with :math:`I` the
intensity and :math:`\gamma_{ab}` the normalized isotropic overlap
reduction function of the native geometry. For an isotropic, unpolarized
population this gives :math:`E[Y_{ab}] = \gamma_{ab} S_h`. The sum is never
divided by the realized count.

Each pair in the network is projected once per chunk in one fused kernel,
giving an array of shape ``(pairs, frequencies)``.
:func:`~astrogwb.statistics.cross_correlation_estimator` weights them into a network
spectrum ``(F,)`` on demand.

The simulator draws independent Poisson counts in segments of exactly
:math:`\tau` seconds and selects distinct rows from the supplied pool. Each
selected merger belongs to one segment; each segment holds its own
coherent strain :math:`h_a^s = \sum_{i \in s} H_{a,i}`. When detector noise
is enabled, each segment, including empty ones, adds independent noise
:math:`n_a^s \sim \mathcal{CN}(0, \tau S_a / 2)` in each detector and bin.
The noise is zero by default. With the segment
cross-power :math:`C_{ab} = \sum_s \mathrm{Re}[(h_a^s + n_a^s)^* (h_b^s + n_b^s)]`
and :math:`r_i = 1 / \sqrt{\tau \Delta f_i}`, the pair spectrum is

.. math::

    Y_{ab} = \frac{5}{T_\mathrm{seg}} \left[ (1 - r_i) \sum_i K_{ab,i} + r_i C_{ab} \right].

:math:`E[C_{ab}] = \sum_i K_{ab,i}`, so the mean stays the incoherent signal;
the zero-mean part (noise by noise, signal by noise, source interference) is
scaled by :math:`r_i`, which makes the noise-by-noise term have the standard
deviation :func:`~astrogwb.detector.pair_cross_power_std`. Every bin must hold a
whole number :math:`\tau \Delta f_i \ge 1` of the segment's Fourier modes, so
that bins are sums of disjoint Fourier modes, e.g. a 1 Hz linear grid with a
power-of-two :math:`\tau`; a log grid does not qualify. The model holds while
signals are mostly shorter than :math:`\tau`. Bins where a PSD is not
finite get no noise. Earth's rotation during a signal is not modelled.

Before coherent summation, both polarizations receive the geocentric arrival
phase :math:`\exp(-2\pi i f t_i)`, with :math:`t_i` sampled in seconds relative
to segment center within :math:`[-\tau/2, \tau/2)`. The simulator draws these
times independently of row selection. Detector projection applies geometric
delays once. Signals remain complete within their assigned segment, without
window truncation or boundary crossing.

A background is stored with :meth:`BackgroundDraw.write` together with the
:class:`~astrogwb.background.simulation.SimulationConfig` it was simulated with.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from astrogwb import __version__
from astrogwb.background.simulation.config import SimulationConfig
from astrogwb.detector import Network, network_pairs
from astrogwb.detector.projection import project_polarizations_fd
from astrogwb.frequency import bin_widths
from astrogwb.populations.io import _SKY_COLUMNS as SKY_COLUMNS
from astrogwb.utils.io import atomic_h5, h5py
from astrogwb.waveform.base import WaveformGenerator

__all__ = [
    "BackgroundDraw",
    "project_background",
    "simulate_background",
]

type Polarizations = Callable[[Mapping[str, ArrayLike]], tuple[jax.Array, jax.Array]]

#: The layout version of the files :meth:`BackgroundDraw.write` writes.
_FORMAT_VERSION = 2

#: Root attributes :meth:`BackgroundDraw.write` owns.
_RESERVED_ATTRS = frozenset({"metadata", "version", "format_version"})

#: :math:`\langle K_{ab} \rangle = I \gamma_{ab}` over this, the factor in
#: :math:`Y_{ab} = 5 D_{ab} / T`.
_PAIR_RESPONSE_DENOMINATOR = 5.0

#: Absolute slack on ``segment_duration * bin width`` being a whole number of
#: Fourier modes, for the rounding of a grid's spacing.
_MODE_TOLERANCE = 1e-6


class BackgroundDraw(NamedTuple):
    """One realized background projected onto a detector network.

    Attributes
    ----------
    frequencies
        The frequency grid in Hz, ``(F,)``.
    pair_spectral_density
        Pair spectra :math:`Y_{ab}` in :math:`\\mathrm{Hz}^{-1}`, with shape
        ``(pairs, frequencies)``. The pair axis follows
        :func:`~astrogwb.detector.network_pairs`.
    segment_counts
        The number of events in each segment, ``(num_segments,)``. The
        projection of :func:`project_background` has one entry for the whole
        observation.
    """

    frequencies: NDArray[np.float64]
    pair_spectral_density: NDArray[np.float64]
    segment_counts: NDArray[np.int64]

    def write(
        self,
        path: str | Path,
        config: SimulationConfig,
        /,
        **attrs: str | float,
    ) -> Path:
        """Write the background and the configuration it was simulated with, atomically.

        The file holds the datasets ``frequencies``, ``segment_counts`` and
        ``pair_spectral_density``. Its root attributes
        are ``metadata``, the configuration as JSON, ``format_version``, the
        integer layout version :meth:`read` keys on, and ``version``, the
        astrogwb version that wrote it (informational), plus any ``attrs``.

        Parameters
        ----------
        path
            File path.
        config
            The configuration the background was simulated with.
        **attrs
            Extra root attributes, written in the same atomic write and read
            back by :meth:`read_attrs`.

        Returns
        -------
        pathlib.Path
            The file written.

        Raises
        ------
        ValueError
            If ``attrs`` uses a reserved key (``metadata``, ``version`` or
            ``format_version``). Nothing is written.
        """
        clashes = sorted(_RESERVED_ATTRS.intersection(attrs))
        if clashes:
            raise ValueError(f"attrs may not use the reserved keys {clashes}")
        with atomic_h5(path) as file:
            file.attrs["metadata"] = json.dumps(config.to_metadata())
            file.attrs["version"] = __version__
            file.attrs["format_version"] = _FORMAT_VERSION
            for name, value in attrs.items():
                file.attrs[name] = value
            file.create_dataset("frequencies", data=self.frequencies)
            file.create_dataset("segment_counts", data=self.segment_counts)
            file.create_dataset(
                "pair_spectral_density", data=self.pair_spectral_density
            )
        return Path(path)

    @classmethod
    def read(cls, path: str | Path) -> tuple[BackgroundDraw, SimulationConfig]:
        """Read a background and its configuration written by :meth:`write`.

        Parameters
        ----------
        path
            File path.

        Returns
        -------
        BackgroundDraw
            The background's pair spectra in canonical order.
        SimulationConfig
            The configuration it was simulated with.

        Raises
        ------
        ValueError
            If the file's ``format_version`` is missing or unsupported.
        """
        with h5py.File(path, "r") as file:
            _check_format(path, file.attrs)
            config = SimulationConfig.from_metadata(json.loads(file.attrs["metadata"]))
            draw = cls(
                np.asarray(file["frequencies"][()], dtype=np.float64),
                np.asarray(file["pair_spectral_density"][()], dtype=np.float64),
                np.asarray(file["segment_counts"][()], dtype=np.int64),
            )
        return draw, config

    @staticmethod
    def read_attrs(path: str | Path) -> dict[str, str | float | int]:
        """The extra root attributes :meth:`write` stored as ``attrs``.

        Parameters
        ----------
        path
            File path.

        Returns
        -------
        dict
            The attributes other than the reserved ``metadata``, ``version``
            and ``format_version``, as Python scalars.

        Raises
        ------
        ValueError
            As for :meth:`read`, if the file's format is not readable.
        """
        with h5py.File(path, "r") as file:
            _check_format(path, file.attrs)
            return {
                name: value.item() if isinstance(value, np.generic) else value
                for name, value in file.attrs.items()
                if name not in _RESERVED_ATTRS
            }


def _check_format(path: str | Path, attrs: Mapping[str, object]) -> None:
    """Raise unless the file's layout is one this version can read."""
    version = attrs.get("version")
    format_version = attrs.get("format_version")
    if format_version == _FORMAT_VERSION:
        return
    raise ValueError(
        f"{path} has format_version {format_version} (written by astrogwb "
        f"{version}); astrogwb {__version__} reads format version {_FORMAT_VERSION}; "
        "regenerate the background"
    )


def _event_columns(
    sources: Mapping[str, ArrayLike], required: Sequence[str] = SKY_COLUMNS
) -> dict[str, NDArray]:
    """The columns of an event pool as host arrays, validated.

    Raises
    ------
    ValueError
        If a ``required`` column is missing, a column is not one-dimensional
        or the columns differ in length, or a column holds a non-finite value.
    """
    columns = {name: np.asarray(values) for name, values in sources.items()}
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"event pool is missing the columns {missing}")
    size = len(next(iter(columns.values()), ()))
    for name, column in columns.items():
        if column.shape != (size,):
            raise ValueError(
                f"event pool column {name!r} has shape {column.shape}; "
                f"every column must have shape ({size},)"
            )
        if not np.isfinite(column).all():
            raise ValueError(f"event pool column {name!r} has non-finite values")
    return columns


def project_background(
    sources: Mapping[str, ArrayLike],
    generator: WaveformGenerator,
    network: Network,
    *,
    observation_seconds: float,
    chunk_size: int = 1024,
) -> BackgroundDraw:
    """Project supplied physical events into incoherent pair spectra, in Hz^-1.

    Each row contributes exactly once. The normalization is the observation
    duration in seconds. No noise is added.

    Parameters
    ----------
    sources
        The events, one column per waveform parameter plus the sky columns
        ``right_ascension``, ``declination`` and ``polarization_angle``, all
        one-dimensional and of the same length.
    generator
        Waveform generator for the supplied sources.
    network
        Detectors keyed by name.
    observation_seconds
        The normalization, in seconds.
    chunk_size
        Waveforms per batch; sets cost, not the result.

    Returns
    -------
    BackgroundDraw
        The spectra, with ``segment_counts`` of one entry, the number of rows.

    Raises
    ------
    ValueError
        If ``observation_seconds`` or ``chunk_size`` is not positive, or the
        pool is invalid (see the event pool columns above).
    """
    if observation_seconds <= 0:
        raise ValueError("observation_seconds must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    jax.config.update("jax_enable_x64", True)
    columns = _event_columns(sources)
    count = len(columns["right_ascension"])
    if count:
        generator.check_sources(columns)
    frequencies = generator.frequencies
    geometry = _network_geometry(network)

    incoherent = jnp.zeros((geometry.num_pairs, frequencies.shape[0]))
    for _, chunk, num_valid in _padded_chunks(columns, chunk_size):
        incoherent += _chunk_pair_power(
            frequencies,
            chunk,
            num_valid,
            geometry,
            polarizations=generator.polarizations,
        )

    spectrum = _PAIR_RESPONSE_DENOMINATOR * np.asarray(incoherent) / observation_seconds
    return BackgroundDraw(np.asarray(frequencies), spectrum, np.array([count]))


def simulate_background(
    key: jax.Array,
    sources: Mapping[str, ArrayLike],
    generator: WaveformGenerator,
    network: Network,
    *,
    poisson_rate: float,
    num_segments: int,
    segment_duration: float,
    chunk_size: int = 1024,
    segment_batch_size: int = 1024,
    simulate_detector_noise: bool = False,
) -> BackgroundDraw:
    """Draw independent Poisson segment counts from a fixed physical event pool.

    Each pool row supplies waveform parameters, right ascension, declination
    and polarization angle. Rows are selected uniformly without replacement
    across the entire draw. Arrival times are drawn uniformly within each
    segment; any catalog coalescence times are replaced.

    The strain is summed coherently in each segment. Detector noise is added
    to each segment when requested. The same key selects the same events and
    arrival times in either mode. Rows follow canonical detector pair order.

    Parameters
    ----------
    key
        Typed PRNG key of the draw.
    sources
        The event pool; see :func:`project_background` for its columns.
    generator
        Waveform generator for the selected sources.
    network
        Detectors keyed by name.
    poisson_rate
        Rate in events per second of the population the pool represents.
    num_segments
        Number of segments.
    segment_duration
        Segment length in seconds.
    chunk_size
        Waveforms per batch; sets cost, not the result.
    segment_batch_size
        Segments per batch; sets cost, not the result.
    simulate_detector_noise
        Add independent detector noise to every segment.

    Returns
    -------
    BackgroundDraw
        The spectra, normalized by ``num_segments * segment_duration``, with
        the event count of each segment.

    Raises
    ------
    ValueError
        If a count, duration, rate or batch size is not valid, the pool is
        invalid, the draw needs more rows than the pool holds, or a bin is not
        a whole number of the segment's Fourier modes.
    """
    if min(chunk_size, segment_batch_size) < 1:
        raise ValueError("chunk and segment batch sizes must be positive")
    jax.config.update("jax_enable_x64", True)
    event_key, noise_key = jax.random.split(key)
    columns, counts = _draw_events(
        event_key,
        _event_columns(sources),
        poisson_rate,
        num_segments,
        segment_duration,
    )
    if counts.any():
        generator.check_sources(columns)
    frequencies = generator.frequencies
    modes = _modes_per_bin(frequencies, segment_duration)
    geometry = _network_geometry(network)
    psds = _network_psds(network, frequencies) if simulate_detector_noise else None

    incoherent, coherent = _segmented_pair_powers(
        columns,
        counts,
        frequencies,
        geometry,
        generator.polarizations,
        noise_key=noise_key,
        psds=psds,
        segment_duration=segment_duration,
        chunk_size=chunk_size,
        segment_batch_size=segment_batch_size,
    )

    # Keep the incoherent mean and scale the zero-mean fluctuations by r_i.
    ratio = 1.0 / np.sqrt(modes)
    blended = (1.0 - ratio) * incoherent + ratio * coherent
    observation_seconds = len(counts) * segment_duration
    spectrum = _PAIR_RESPONSE_DENOMINATOR * blended / observation_seconds
    return BackgroundDraw(np.asarray(frequencies), spectrum, counts)


def _draw_events(
    key: jax.Array,
    pool: Mapping[str, NDArray],
    poisson_rate: float,
    num_segments: int,
    segment_duration: float,
) -> tuple[dict[str, NDArray], NDArray[np.int64]]:
    """Select the rows of every segment and give each an arrival time.

    Returns the selected rows, ordered by segment, with a ``coa_time`` column
    in seconds from the segment center, and the event count of each segment.
    """
    if not np.isfinite(poisson_rate) or poisson_rate < 0:
        raise ValueError("poisson_rate must be finite and nonnegative")
    if num_segments < 1:
        raise ValueError(f"num_segments must be positive; got {num_segments}")
    if not np.isfinite(segment_duration) or segment_duration <= 0:
        raise ValueError(f"segment_duration must be positive; got {segment_duration}")
    # Row selection runs on host arrays. A varying event count need not
    # compile a new random kernel for every realization.
    rng = np.random.default_rng(np.asarray(jax.random.key_data(key)))
    counts = rng.poisson(poisson_rate * segment_duration, num_segments)
    count = int(counts.sum())
    size = len(next(iter(pool.values()), ()))
    if count > size:
        raise ValueError(
            f"event pool has {size} rows but this draw requires {count} "
            f"(expected {poisson_rate * segment_duration * num_segments:.1f}); "
            "supply a larger pool or lower the rate"
        )
    indices = rng.choice(size, size=count, replace=False)
    selected = {name: column[indices] for name, column in pool.items()}
    selected["coa_time"] = rng.uniform(
        -segment_duration / 2, segment_duration / 2, count
    )
    return selected, counts


class _Geometry(NamedTuple):
    """The detector axis and the pair axis over it, as the kernels consume them.

    Detectors are in sorted-name order: ``tensors`` is ``(D, 3, 3)`` and
    ``locations`` ``(D, 3)``. Pair ``p`` correlates detectors ``first[p]`` and
    ``second[p]``, pairs following :func:`~astrogwb.detector.network_pairs`.
    """

    tensors: jax.Array
    locations: jax.Array
    first: jax.Array
    second: jax.Array

    @property
    def num_pairs(self) -> int:
        return self.first.shape[0]


def _network_geometry(network: Network) -> _Geometry:
    names = sorted(network)
    index = {name: position for position, name in enumerate(names)}
    pairs = network_pairs(network)
    return _Geometry(
        tensors=jnp.stack([network[name].geometry.response_tensor for name in names]),
        locations=jnp.stack([network[name].geometry.location for name in names]),
        first=jnp.array([index[first] for first, _ in pairs], dtype=int),
        second=jnp.array([index[second] for _, second in pairs], dtype=int),
    )


def _network_psds(network: Network, frequencies: ArrayLike) -> jax.Array:
    """The ``(D, F)`` noise PSDs in the detector order of :func:`_network_geometry`."""
    frequencies = np.asarray(frequencies)
    return jnp.asarray(
        np.stack([network[name].sensitivity(frequencies) for name in sorted(network)])
    )


def _modes_per_bin(frequencies: ArrayLike, segment_duration: float) -> NDArray:
    """The ``(F,)`` count of a segment's Fourier modes in each frequency bin.

    Raises
    ------
    ValueError
        If a bin holds fewer than one or a fractional number of modes.
    """
    modes = segment_duration * np.asarray(bin_widths(frequencies), dtype=np.float64)
    whole = np.allclose(modes, np.round(modes), rtol=0.0, atol=_MODE_TOLERANCE)
    if modes.min() < 1.0 or not whole:
        raise ValueError(
            f"every bin must hold a whole number of the {segment_duration} s "
            "segment's Fourier modes: segment_duration * bin width must be an "
            "integer of at least 1, e.g. a 1 Hz linear grid with a power-of-two "
            "segment"
        )
    return modes


def _pad(values: NDArray, size: int) -> NDArray:
    """``values`` extended to length ``size`` by repeating its first entry."""
    return np.concatenate([values, np.repeat(values[:1], size - len(values))])


def _padded_chunks(
    columns: Mapping[str, NDArray], size: int
) -> Iterator[tuple[int, dict[str, NDArray], int]]:
    """``(start, rows, num_valid)`` for consecutive chunks of ``size`` rows.

    A Poisson count differs per realization and every distinct array shape is
    a compile, so the kernels only ever see chunks of one static ``size``: the
    last chunk is padded. Padding repeats the chunk's first row, a physical
    source, so its waveform stays finite; the kernels zero the strain of rows
    from ``num_valid`` on.
    """
    count = len(next(iter(columns.values()), ()))
    for start in range(0, count, size):
        rows = {
            name: _pad(column[start : start + size], size)
            for name, column in columns.items()
        }
        yield start, rows, min(size, count - start)


def _detector_strain(
    polarizations: Polarizations,
    frequencies: jax.Array,
    sources: Mapping[str, jax.Array],
    num_valid: jax.Array,
    geometry: _Geometry,
) -> jax.Array:
    """The ``(n, D, F)`` complex strain of a padded chunk; padding rows are zero."""
    plus, cross = polarizations(sources)
    real = (jnp.arange(plus.shape[0]) < num_valid)[:, None]
    plus = jnp.where(real, plus, 0.0)
    cross = jnp.where(real, cross, 0.0)

    def project(tensor: jax.Array, location: jax.Array) -> jax.Array:
        return project_polarizations_fd(
            tensor,
            location,
            frequencies,
            plus,
            cross,
            0.0,
            right_ascension=sources["right_ascension"],
            declination=sources["declination"],
            polarization_angle=sources["polarization_angle"],
        )

    return jax.vmap(project, out_axes=1)(geometry.tensors, geometry.locations)


def _pair_power(strain: jax.Array, geometry: _Geometry) -> jax.Array:
    """The ``(P, F)`` cross-power of an ``(N, D, F)`` strain, summed over ``N``.

    The real part of :math:`h_a^* h_b` for each pair :math:`(a, b)`.
    """
    first = strain[:, geometry.first]
    second = strain[:, geometry.second]
    return jnp.real(jnp.conj(first) * second).sum(axis=0)


@partial(jax.jit, static_argnames="polarizations")
def _chunk_pair_power(
    frequencies: jax.Array,
    sources: Mapping[str, jax.Array],
    num_valid: jax.Array,
    geometry: _Geometry,
    *,
    polarizations: Polarizations,
) -> jax.Array:
    """The ``(P, F)`` incoherent cross-power :math:`\\sum_i K_{ab,i}` of a chunk."""
    strain = _detector_strain(polarizations, frequencies, sources, num_valid, geometry)
    return _pair_power(strain, geometry)


@partial(jax.jit, static_argnames=("polarizations", "num_segments"))
def _chunk_pair_power_and_segment_strain(
    frequencies: jax.Array,
    sources: Mapping[str, jax.Array],
    num_valid: jax.Array,
    segment_ids: jax.Array,
    geometry: _Geometry,
    *,
    polarizations: Polarizations,
    num_segments: int,
) -> tuple[jax.Array, jax.Array]:
    """The incoherent cross-power of a chunk and its strain summed per segment.

    Returns the ``(P, F)`` cross-power and the ``(num_segments, D, F)``
    strain, ``segment_ids`` holding the segment of each row.
    """
    strain = _detector_strain(polarizations, frequencies, sources, num_valid, geometry)
    per_segment = jax.ops.segment_sum(strain, segment_ids, num_segments=num_segments)
    return _pair_power(strain, geometry), per_segment


@jax.jit
def _segment_batch_power(
    noise_key: jax.Array,
    segment_ids: jax.Array,
    strain: jax.Array,
    psds: jax.Array | None,
    geometry: _Geometry,
    segment_duration: jax.Array | float,
    num_segments: jax.Array | int,
) -> jax.Array:
    """The ``(P, F)`` cross-power of signal plus noise, summed over a batch of segments.

    ``segment_ids`` are the global indices of the batch, which key its noise,
    and ``strain`` its ``(B, D, F)`` coherent signal. ``psds`` are the
    optional ``(D, F)`` noise curves; non-finite bins get no noise. Segments
    at or beyond ``num_segments``, the padding of the last batch, add nothing.
    """
    if psds is not None:
        amplitude = jnp.sqrt(
            segment_duration * jnp.where(jnp.isfinite(psds), psds, 0.0) / 4.0
        )

        def noise(segment: jax.Array) -> jax.Array:
            real_key, imag_key = jax.random.split(
                jax.random.fold_in(noise_key, segment)
            )
            return amplitude * jax.lax.complex(
                jax.random.normal(real_key, amplitude.shape),
                jax.random.normal(imag_key, amplitude.shape),
            )

        strain = strain + jax.vmap(noise)(segment_ids)
    real = (segment_ids < num_segments)[:, None, None]
    return _pair_power(jnp.where(real, strain, 0.0), geometry)


def _segmented_pair_powers(
    columns: Mapping[str, NDArray],
    counts: NDArray[np.int64],
    frequencies: jax.Array,
    geometry: _Geometry,
    polarizations: Polarizations,
    *,
    noise_key: jax.Array,
    psds: jax.Array | None,
    segment_duration: float,
    chunk_size: int,
    segment_batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The incoherent and coherent ``(P, F)`` cross-powers of a segmented draw.

    The incoherent power is :math:`\\sum_i K_{ab,i}` and the coherent power
    :math:`C_{ab}`, the cross-power of each segment's summed strain plus
    noise, summed over segments. ``columns`` are ordered by segment, segment
    ``s`` holding ``counts[s]`` consecutive rows.

    Segments are processed ``segment_batch_size`` at a time, and the rows of a
    batch ``chunk_size`` at a time, so that both kernels compile once.
    """
    num_segments = len(counts)
    batch_size = min(segment_batch_size, num_segments)
    segment_of_row = np.repeat(np.arange(num_segments), counts)
    first_row = np.concatenate([[0], np.cumsum(counts)])
    num_detectors = geometry.tensors.shape[0]
    num_frequencies = frequencies.shape[0]

    incoherent = jnp.zeros((geometry.num_pairs, num_frequencies))
    coherent = jnp.zeros_like(incoherent)
    for low in range(0, num_segments, batch_size):
        high = min(low + batch_size, num_segments)
        rows = slice(first_row[low], first_row[high])
        batch_columns = {name: column[rows] for name, column in columns.items()}
        batch_segment = segment_of_row[rows] - low

        strain = jnp.zeros((batch_size, num_detectors, num_frequencies), complex)
        for start, chunk, num_valid in _padded_chunks(batch_columns, chunk_size):
            chunk_power, chunk_strain = _chunk_pair_power_and_segment_strain(
                frequencies,
                chunk,
                num_valid,
                _pad(batch_segment[start : start + chunk_size], chunk_size),
                geometry,
                polarizations=polarizations,
                num_segments=batch_size,
            )
            incoherent += chunk_power
            strain += chunk_strain

        coherent += _segment_batch_power(
            noise_key,
            jnp.arange(low, low + batch_size),
            strain,
            psds,
            geometry,
            segment_duration,
            num_segments,
        )
    return np.asarray(incoherent), np.asarray(coherent)
