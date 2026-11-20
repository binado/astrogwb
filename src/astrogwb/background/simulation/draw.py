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

:math:`\phi_{ab,i}` the inter-site delay phase. Sources are summed without
cross terms (no source interference), :math:`D_{ab} = \sum_i K_{ab,i}`, and
normalized by the observation time :math:`T` as

.. math::

    Y_{ab} = \frac{5\, D_{ab}}{T}.

Over an isotropic sky and uniform polarization angle
:math:`\langle K_{ab} \rangle = I \gamma_{ab} / 5`, with :math:`I` the
intensity and :math:`\gamma_{ab}` the normalized isotropic overlap
reduction function of the native geometry. For an isotropic, unpolarized
population this gives :math:`E[Y_{ab}] = \gamma_{ab} S_h`. The sum is never
divided by the realized count.

Each pair requested by at least one network is projected once per chunk in
one fused kernel, and each network keeps its own rows, ``(P_n, F)``.
:func:`~astrogwb.statistics.cross_correlation_estimator` weights them into a network
spectrum ``(F,)`` on demand.

The simulator draws independent Poisson counts in segments of exactly
:math:`\tau` seconds and selects distinct rows from the supplied pool. Each
selected merger belongs to one segment; each segment holds its own
coherent strain :math:`h_a^s = \sum_{i \in s} H_{a,i}` and fresh independent
noise :math:`n_a^s \sim \mathcal{CN}(0, \tau S_a / 2)` in each
detector and bin; empty segments still carry noise. With the segment
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
import re
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from astrogwb import __version__
from astrogwb.background.simulation._chunks import iter_rows, pad_rows
from astrogwb.background.simulation.config import SimulationConfig, _network_detectors
from astrogwb.detector import Networks, network_pairs
from astrogwb.detector.projection import project_polarizations_fd
from astrogwb.frequency import bin_widths
from astrogwb.utils.io import atomic_h5, h5py
from astrogwb.waveform.base import WaveformGenerator
from astrogwb.waveform.factory import waveform_generator
from astrogwb.waveform.metadata import WaveformConfig

__all__ = [
    "BackgroundDraw",
    "project_background",
    "simulate_background",
]

#: The layout version of the files :meth:`BackgroundDraw.write` writes.
_FORMAT_VERSION = 1

#: The layout versions :meth:`BackgroundDraw.read` reads.
_READABLE_FORMATS = frozenset({_FORMAT_VERSION})

#: Root attributes :meth:`BackgroundDraw.write` owns.
_RESERVED_ATTRS = frozenset({"metadata", "version", "format_version"})

#: Files written before ``format_version`` existed, which were layout 1.
_LEGACY_VERSION = re.compile(r"0\.8\.")

#: The columns every event pool must supply besides its waveform parameters.
SKY_COLUMNS = ("right_ascension", "declination", "polarization_angle")


#: :math:`\langle K_{ab} \rangle = I \gamma_{ab}` over this, the factor in
#: :math:`Y_{ab} = 5 D_{ab} / T`.
_PAIR_RESPONSE_DENOMINATOR = 5.0

#: Absolute slack on ``segment_duration * bin width`` being a whole number of
#: Fourier modes, for the rounding of a grid's spacing.
_MODE_TOLERANCE = 1e-6


class _Geometry(NamedTuple):
    """The joint detector and pair axes, as the kernels consume them.

    ``tensors`` ``(D, 3, 3)``, ``locations`` ``(D, 3)`` and ``psds`` ``(D, F)``
    cover the union of the networks' detectors. ``pairs`` indexes only pairs
    requested by a network; ``rows[name]`` selects the rows for network ``name``
    in the order of :func:`~astrogwb.detector.network_pairs`.
    """

    tensors: NDArray[np.float64]
    locations: NDArray[np.float64]
    psds: NDArray[np.float64]
    pairs: tuple[tuple[int, int], ...]
    rows: dict[str, list[int]]


def _network_geometry(
    networks: Networks, frequencies: NDArray[np.float64]
) -> _Geometry:
    """Index the requested pairs over the shared detector axis."""
    definitions = _network_detectors(networks)
    names = sorted(definitions)
    detector_index = {name: index for index, name in enumerate(names)}
    detectors = tuple(definitions[name] for name in names)
    pairs_by_network = {
        name: network_pairs(network) for name, network in sorted(networks.items())
    }
    pairs = sorted({pair for members in pairs_by_network.values() for pair in members})
    rows = {pair: row for row, pair in enumerate(pairs)}
    return _Geometry(
        tensors=np.stack([det.geometry.response_tensor for det in detectors]),
        locations=np.stack([det.geometry.location for det in detectors]),
        psds=np.stack([np.asarray(det.sensitivity(frequencies)) for det in detectors]),
        pairs=tuple(
            (detector_index[first], detector_index[second]) for first, second in pairs
        ),
        rows={
            name: [rows[pair] for pair in members]
            for name, members in pairs_by_network.items()
        },
    )


class _Kernels(NamedTuple):
    """The generator of a waveform and the jitted chunk kernels built on it."""

    generator: WaveformGenerator
    pair_sums: Callable[..., jax.Array]
    segment_sums: Callable[..., tuple[jax.Array, jax.Array]]


@cache
def _kernels(waveform: WaveformConfig, pairs: tuple[tuple[int, int], ...]) -> _Kernels:
    """The chunk kernels of ``waveform`` over detector ``pairs``.

    Cached on its hashable arguments, so every background of one waveform and
    pair layout shares one compile per chunk shape. The geometry is an
    argument of the kernels, not a constant of them.
    """
    generator = waveform_generator(waveform)

    def strains(
        frequencies: jax.Array,
        sources: Mapping[str, jax.Array],
        valid: jax.Array,
        tensors: jax.Array,
        locations: jax.Array,
    ) -> jax.Array:
        """The ``(D, n, F)`` complex detector strains of one padded chunk.

        Padding rows have zero strain.
        """
        plus, cross = generator.polarizations(sources)
        real = (jnp.arange(plus.shape[0]) < valid)[:, None]
        plus = jnp.where(real, plus, 0.0)
        cross = jnp.where(real, cross, 0.0)
        return jax.vmap(
            lambda tensor, location: project_polarizations_fd(
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
        )(tensors, locations)

    def pair_cross(strain: jax.Array) -> jax.Array:
        """The ``(P_union, F)`` real cross-powers summed over the sources."""
        return jnp.stack(
            [
                jnp.real(jnp.conj(strain[first]) * strain[second]).sum(axis=0)
                for first, second in pairs
            ]
        )

    def pair_sums(
        frequencies: jax.Array,
        sources: Mapping[str, jax.Array],
        valid: jax.Array,
        tensors: jax.Array,
        locations: jax.Array,
    ) -> jax.Array:
        """The ``(P_union, F)`` cross-power summed over one padded chunk's sources."""
        return pair_cross(strains(frequencies, sources, valid, tensors, locations))

    def segment_sums(
        frequencies: jax.Array,
        sources: Mapping[str, jax.Array],
        valid: jax.Array,
        local_ids: jax.Array,
        tensors: jax.Array,
        locations: jax.Array,
        *,
        num_segments: int,
    ) -> tuple[jax.Array, jax.Array]:
        """The incoherent cross-power and per-segment strain of one padded chunk.

        Returns the ``(P_union, F)`` real cross-powers and the
        ``(num_segments, D, F)`` complex strain summed over the sources of each
        segment, ``local_ids`` being the segment of each row relative to the
        batch.
        """
        strain = strains(frequencies, sources, valid, tensors, locations)
        coherent = jax.ops.segment_sum(
            jnp.moveaxis(strain, 1, 0), local_ids, num_segments=num_segments
        )
        return pair_cross(strain), coherent

    return _Kernels(
        generator=generator,
        pair_sums=jax.jit(pair_sums),
        segment_sums=jax.jit(segment_sums, static_argnames="num_segments"),
    )


class BackgroundDraw(NamedTuple):
    """One realized background projected onto the requested networks.

    Attributes
    ----------
    frequencies
        The frequency grid in Hz, ``(F,)``.
    spectra
        Each network's pair spectra :math:`Y_{ab}` in :math:`\\mathrm{Hz}^{-1}`,
        ``(P_n, F)``, keyed by network name. The pair axis follows
        :func:`~astrogwb.detector.network_pairs`.
    segment_counts
        The number of events in each segment, ``(num_segments,)``. The
        projection of :func:`project_background` has one entry for the whole
        observation.
    """

    frequencies: NDArray[np.float64]
    spectra: dict[str, NDArray[np.float64]]
    segment_counts: NDArray[np.int64]

    def write(
        self,
        path: str | Path,
        config: SimulationConfig,
        /,
        **attrs: str | float,
    ) -> Path:
        """Write the background and the configuration it was simulated with, atomically.

        The file holds the datasets ``frequencies``, ``segment_counts`` and, per
        network, ``networks/<name>/pair_spectral_density``. Its root attributes
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
            for name, spectrum in self.spectra.items():
                group = file.create_group(f"networks/{name}")
                group.create_dataset("pair_spectral_density", data=spectrum)
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
            The background, networks in sorted order.
        SimulationConfig
            The configuration it was simulated with.

        Raises
        ------
        ValueError
            If the file's ``format_version`` is not one this version reads. A
            file without ``format_version`` is read as format 1 when its
            ``version`` is ``0.8.*``, as written before format versions were
            recorded.
        """
        with h5py.File(path, "r") as file:
            _check_format(path, file.attrs)
            config = SimulationConfig.from_metadata(json.loads(file.attrs["metadata"]))
            stored = file["networks"]
            draw = cls(
                np.asarray(file["frequencies"][()], dtype=np.float64),
                {
                    name: np.asarray(
                        stored[name]["pair_spectral_density"][()], dtype=np.float64
                    )
                    for name in sorted(stored)
                },
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
    if format_version is None and isinstance(version, str):
        if _LEGACY_VERSION.match(version):
            return
    elif format_version in _READABLE_FORMATS:
        return
    raise ValueError(
        f"{path} has format_version {format_version} (written by astrogwb "
        f"{version}); astrogwb {__version__} reads format versions "
        f"{sorted(_READABLE_FORMATS)} and files written by astrogwb 0.8.*; "
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


def _projection(
    waveform: WaveformConfig, networks: Networks
) -> tuple[jax.Array, _Geometry, _Kernels]:
    jax.config.update("jax_enable_x64", True)
    frequencies = np.asarray(waveform_generator(waveform).frequencies)
    geometry = _network_geometry(networks, frequencies)
    return jnp.asarray(frequencies), geometry, _kernels(waveform, geometry.pairs)


def project_background(
    sources: Mapping[str, ArrayLike],
    waveform: WaveformConfig,
    networks: Networks,
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
    waveform
        Waveform generator settings.
    networks
        Networks keyed by name, each mapping detector names to detectors.
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
    grid, geometry, kernels = _projection(waveform, networks)
    columns = _event_columns(sources)
    count = len(columns["right_ascension"])
    if count:
        kernels.generator.check_sources(columns)
    tensors = jnp.asarray(geometry.tensors)
    locations = jnp.asarray(geometry.locations)
    total = jnp.zeros((len(geometry.pairs), grid.size))
    for start, _ in iter_rows(count, chunk_size):
        rows, valid = pad_rows(columns, start, chunk_size)
        total += kernels.pair_sums(grid, rows, jnp.asarray(valid), tensors, locations)
    union = _PAIR_RESPONSE_DENOMINATOR * np.asarray(total) / observation_seconds
    return BackgroundDraw(
        np.asarray(grid),
        {k: union[v] for k, v in geometry.rows.items()},
        np.array([count]),
    )


def _draw_events(
    key: jax.Array,
    sources: Mapping[str, ArrayLike],
    observer_rate_hz: float,
    num_segments: int,
    segment_seconds: float,
) -> tuple[dict[str, NDArray], NDArray[np.int64]]:
    if not np.isfinite(observer_rate_hz) or observer_rate_hz < 0:
        raise ValueError("observer_rate_hz must be finite and nonnegative")
    if num_segments < 1:
        raise ValueError(f"num_segments must be positive; got {num_segments}")
    if not np.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError(f"segment_seconds must be positive; got {segment_seconds}")
    pool = _event_columns(sources, required=())
    # Row selection runs on host arrays. A varying event count need not
    # compile a new random kernel for every realization.
    rng = np.random.default_rng(np.asarray(jax.random.key_data(key)))
    counts = rng.poisson(observer_rate_hz * segment_seconds, num_segments)
    count = int(counts.sum())
    size = len(next(iter(pool.values()), ()))
    if count > size:
        raise ValueError(
            f"event pool has {size} rows but this draw requires {count} "
            f"(expected {observer_rate_hz * segment_seconds * num_segments:.1f}); "
            "supply a larger pool or lower the rate"
        )
    indices = rng.choice(size, size=count, replace=False)
    selected = {name: col[indices] for name, col in pool.items()}
    selected["coa_time"] = rng.uniform(-segment_seconds / 2, segment_seconds / 2, count)
    return selected, counts


def simulate_background(
    key: jax.Array,
    sources: Mapping[str, ArrayLike],
    waveform: WaveformConfig,
    networks: Networks,
    *,
    observer_rate_hz: float,
    num_segments: int,
    segment_seconds: float,
    chunk_size: int = 1024,
    segment_batch_size: int = 1024,
    incoherent: bool = False,
) -> BackgroundDraw:
    """Draw independent Poisson segment counts from a fixed physical event pool.

    Each pool row supplies waveform parameters, right ascension, declination
    and polarization angle. Rows are selected uniformly without replacement
    across the entire draw. Arrival times are drawn uniformly within each
    segment; any catalog coalescence times are replaced.

    The default sums coherent strain and detector noise in each segment.
    With ``incoherent=True``, it sums individual event powers over the same
    observation: the noiseless expected-signal spectrum of the same selection,
    with no segment interference and no whole-mode requirement on the grid.
    Each network has one row per detector pair in canonical order. The same key
    gives the same draw.

    Parameters
    ----------
    key
        Typed PRNG key of the draw.
    sources
        The event pool; see :func:`project_background` for its columns.
    waveform
        Waveform generator settings.
    networks
        Networks keyed by name that share one event and detector-noise realization.
    observer_rate_hz
        Rate in events per second of the population the pool represents.
    num_segments
        Number of segments.
    segment_seconds
        Segment length in seconds.
    chunk_size
        Waveforms per batch; sets cost, not the result.
    segment_batch_size
        Segments per batch in the coherent path; sets cost, not the result.
    incoherent
        Skip noise and coherent summation.

    Returns
    -------
    BackgroundDraw
        The spectra, normalized by ``num_segments * segment_seconds``, with
        the event count of each segment.

    Raises
    ------
    ValueError
        If a count, duration, rate or batch size is not valid, the pool is
        invalid, the draw needs more rows than the pool holds, or (coherent
        path) a bin is not a whole number of the segment's Fourier modes.
    """
    if min(chunk_size, segment_batch_size) < 1:
        raise ValueError("chunk and segment batch sizes must be positive")
    jax.config.update("jax_enable_x64", True)
    event_key, noise_key = jax.random.split(key)
    columns, counts = _draw_events(
        event_key,
        _event_columns(sources),
        observer_rate_hz,
        num_segments,
        segment_seconds,
    )
    if incoherent:
        result = project_background(
            columns,
            waveform,
            networks,
            observation_seconds=num_segments * segment_seconds,
            chunk_size=chunk_size,
        )
        return result._replace(segment_counts=counts)
    grid, geometry, kernels = _projection(waveform, networks)
    if counts.any():
        kernels.generator.check_sources(columns)
    union = _segmented(
        columns,
        counts,
        noise_key,
        grid,
        geometry,
        kernels,
        chunk_size,
        segment_seconds,
        segment_batch_size,
    )
    return BackgroundDraw(
        np.asarray(grid), {k: union[v] for k, v in geometry.rows.items()}, counts
    )


def _segmented(
    columns: Mapping[str, NDArray],
    counts: NDArray[np.int64],
    noise_key: jax.Array,
    grid: jax.Array,
    geometry: _Geometry,
    kernels: _Kernels,
    chunk_size: int,
    segment_duration: float,
    segment_batch_size: int,
) -> NDArray[np.float64]:
    """The pair spectra with per-segment sources and detector noise."""
    segments = len(counts)
    seconds = segments * segment_duration
    widths = np.asarray(bin_widths(grid), dtype=np.float64)
    modes = segment_duration * widths
    if modes.min() < 1.0 or not np.allclose(
        modes, np.round(modes), rtol=0.0, atol=_MODE_TOLERANCE
    ):
        raise ValueError(
            f"every bin must hold a whole number of the {segment_duration} s "
            "segment's Fourier modes: segment_duration * bin width must be an "
            "integer of at least 1, e.g. a 1 Hz linear grid with a power-of-two "
            "segment"
        )
    labels = np.repeat(np.arange(segments), counts)
    first, second = (jnp.asarray(index) for index in zip(*geometry.pairs, strict=True))
    psds = jnp.asarray(geometry.psds)
    tensors = jnp.asarray(geometry.tensors)
    locations = jnp.asarray(geometry.locations)
    batch = min(segment_batch_size, segments)
    incoherent = jnp.zeros((len(geometry.pairs), grid.shape[0]))
    coherent = jnp.zeros((len(geometry.pairs), grid.shape[0]))
    for low in range(0, segments, batch):
        lo, hi = np.searchsorted(labels, [low, low + batch])
        rows = {name: values[lo:hi] for name, values in columns.items()}
        strain = jnp.zeros((batch, geometry.psds.shape[0], grid.shape[0]), complex)
        for start, _ in iter_rows(int(hi - lo), chunk_size):
            chunk, valid = pad_rows(rows, start, chunk_size)
            ids = np.zeros(chunk_size, dtype=np.int32)
            ids[:valid] = labels[lo + start : lo + start + valid] - low
            chunk_total, chunk_strain = kernels.segment_sums(
                grid,
                chunk,
                jnp.asarray(valid),
                jnp.asarray(ids),
                tensors,
                locations,
                num_segments=batch,
            )
            incoherent = incoherent + chunk_total
            strain = strain + chunk_strain
        coherent = coherent + _segment_power(
            noise_key,
            jnp.arange(low, low + batch),
            strain,
            psds,
            first,
            second,
            segment_duration,
            segments,
        )
    ratio = 1.0 / np.sqrt(modes)
    return (
        _PAIR_RESPONSE_DENOMINATOR
        / seconds
        * (
            (1.0 - ratio) * np.asarray(incoherent, dtype=np.float64)
            + ratio * np.asarray(coherent, dtype=np.float64)
        )
    )


@jax.jit
def _segment_power(
    noise_key: jax.Array,
    segment_ids: jax.Array,
    strain: jax.Array,
    psds: jax.Array,
    first: jax.Array,
    second: jax.Array,
    segment_seconds: jax.Array | float,
    num_segments: jax.Array | int,
) -> jax.Array:
    """The cross-power of signal plus noise summed over a batch of segments, ``(P, F)``.

    ``segment_ids`` are the global segment indices of the batch, which key
    its noise, ``strain`` the ``(B, D, F)`` coherent signal strain of each,
    ``psds`` the ``(D, F)`` noise curves and ``first``, ``second`` the
    detector indices of each pair. Segments at or beyond ``num_segments``
    (the padding of the last batch) add nothing.
    """
    amplitude = jnp.sqrt(
        segment_seconds * jnp.where(jnp.isfinite(psds), psds, 0.0) / 4.0
    )

    def draw(segment: jax.Array) -> jax.Array:
        real_key, imag_key = jax.random.split(jax.random.fold_in(noise_key, segment))
        return amplitude * jax.lax.complex(
            jax.random.normal(real_key, psds.shape),
            jax.random.normal(imag_key, psds.shape),
        )

    total = strain + jax.vmap(draw)(segment_ids)
    power = jnp.real(jnp.conj(total[:, first]) * total[:, second])
    real = (segment_ids < num_segments)[:, None, None]
    return jnp.where(real, power, 0.0).sum(axis=0)
