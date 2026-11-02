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
polarization power and :math:`\gamma_{ab}` the normalized isotropic overlap
reduction function of the native geometry. For an isotropic, unpolarized
population this gives :math:`E[Y_{ab}] = \gamma_{ab} S_h`. The sum is never
divided by the realized count.

Every pair of the union of the networks' detectors is projected once per chunk
in one fused kernel, and each network keeps its own rows, ``(P_n, F)``.
:func:`combine_pair_spectra` weights them into a network spectrum ``(F,)`` on
demand.

Without a ``segment_duration`` the background is deterministic given its
inputs and carries no detector noise. With one, :math:`\tau`, the observation is
cut into :math:`N_\mathrm{seg} = \lfloor T/\tau \rfloor` segments of length
:math:`\tau_\mathrm{eff} = T / N_\mathrm{seg}`. Sources are Poisson in time, so
each is assigned to a uniformly drawn segment :math:`s`, which holds its own
coherent strain :math:`h_a^s = \sum_{i \in s} H_{a,i}` and fresh independent
noise :math:`n_a^s \sim \mathcal{CN}(0, \tau_\mathrm{eff} S_a / 2)` in each
detector and bin; empty segments still carry noise. With the segment
cross-power :math:`C_{ab} = \sum_s \mathrm{Re}[(h_a^s + n_a^s)^* (h_b^s + n_b^s)]`
and :math:`r_i = 1 / \sqrt{\tau_\mathrm{eff} \Delta f_i}`, the pair spectrum is

.. math::

    Y_{ab} = \frac{5}{T} \left[ (1 - r_i) \sum_i K_{ab,i} + r_i C_{ab} \right].

:math:`E[C_{ab}] = \sum_i K_{ab,i}`, so the mean stays the incoherent signal;
the zero-mean part (noise by noise, signal by noise, source interference) is
scaled by :math:`r_i`, which makes the noise-by-noise term have the standard
deviation :func:`~astrogwb.detector.pair_bin_scale` for any grid and any
:math:`\tau`. The model holds while signals are mostly shorter than
:math:`\tau` and :math:`\tau_\mathrm{eff} \Delta f_i \ge 1`. Segment labels
and noise are folded from the realization's key (streams
:data:`SEGMENT_STREAM` and :data:`NOISE_STREAM`). Bins where a PSD is not
finite get no noise. Earth's rotation during a signal is not modelled.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb import __version__
from astrogwb.detector import (
    effective_psd,
    gaussian_bin_scale,
    pair_bin_scale,
    pair_terms,
)
from astrogwb.detector.config import DetectorRegistry
from astrogwb.detector.projection import project_polarizations_fd
from astrogwb.frequency import bin_widths
from astrogwb.simulators._chunks import iter_rows, pad_rows
from astrogwb.simulators.config import SimulationConfig
from astrogwb.simulators.population import PopulationArtifact
from astrogwb.utils import years_to_seconds
from astrogwb.utils.io import atomic_h5, h5py
from astrogwb.waveform.factory import waveform_generator
from astrogwb.waveform.metadata import WaveformMetadata

if TYPE_CHECKING:
    from astrogwb.waveform import WaveformGenerator

__all__ = [
    "NOISE_STREAM",
    "SEGMENT_STREAM",
    "BackgroundArtifact",
    "BackgroundSimulator",
    "NetworkNoise",
    "combine_pair_spectra",
    "select_networks",
]


#: :math:`\langle K_{ab} \rangle = I \gamma_{ab}` over this, the factor in
#: :math:`Y_{ab} = 5 D_{ab} / T`.
PAIR_RESPONSE_DENOMINATOR = 5.0

#: The stream folded into a realization's key for the detector noise.
NOISE_STREAM = 1

#: The stream folded into a realization's key for the segment of each source.
SEGMENT_STREAM = 2


class NetworkNoise(NamedTuple):
    """One network's noise on the background's grid, the likelihood inputs.

    ``pair_overlap`` is the isotropic ORF :math:`\\gamma_{ab}` of each pair,
    ``(P_n, F)``; ``pair_scale`` the per-pair Gaussian scale of
    :func:`~astrogwb.detector.pair_bin_scale`, ``(P_n, F)``; ``effective_psd``
    the network's :func:`~astrogwb.detector.effective_psd`, ``(F,)``; ``scale``
    its :func:`~astrogwb.detector.gaussian_bin_scale`, ``(F,)``. The pairs are
    in the order of :meth:`DetectorRegistry.pairs`.
    """

    pair_overlap: NDArray[np.float64]
    pair_scale: NDArray[np.float64]
    effective_psd: NDArray[np.float64]
    scale: NDArray[np.float64]


def select_networks(
    registry: DetectorRegistry, names: Sequence[str] | None = None
) -> tuple[str, ...]:
    """The sorted network names ``names`` (default all) of ``registry``.

    A network needs at least two detectors to have a pair.

    Parameters
    ----------
    registry
        Detector registry.
    names
        Names to select.

    Raises
    ------
    ValueError
        If the registry defines no networks, or a selected network has fewer
        than two detectors.
    """
    selected = tuple(sorted(registry.networks if names is None else names))
    if not selected:
        raise ValueError(
            "a background needs at least one network; the registry has none"
        )
    for name in selected:
        members = registry.networks[name]
        if len(members) < 2:
            raise ValueError(f"network {name!r} must contain at least two detectors")
    return selected


def used_detectors(registry: DetectorRegistry, names: Sequence[str]) -> tuple[str, ...]:
    """The sorted detectors that appear in the networks ``names`` of ``registry``.

    Parameters
    ----------
    registry
        Detector registry.
    names
        Names to select.
    """
    return tuple(sorted(registry.select(names).detectors))


@dataclass(frozen=True)
class BackgroundArtifact:
    """The per-network spectra of one simulation, in memory or read from disk.

    ``spectra`` holds each network's pair spectra :math:`Y_{ab}` in
    :math:`\\mathrm{Hz}^{-1}`, ``(P_n, F)``, keyed by network name in sorted
    order; the pair axis follows :meth:`pairs`. ``config`` is the
    :class:`~astrogwb.simulators.SimulationConfig` it was simulated from and
    ``rng_key`` the data of the key it was drawn at. ``count``,
    ``total_merger_rate`` (per second) and ``hyperparameters`` are the
    realization's, carried over from its population.
    """

    frequencies: NDArray[np.float64]
    spectra: dict[str, NDArray[np.float64]]
    config: SimulationConfig
    rng_key: tuple[int, ...]
    count: int
    total_merger_rate: float
    hyperparameters: dict[str, float]

    @property
    def pairs(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """The detector pairs of each network's pair axis."""
        return {name: tuple(self.config.detectors.pairs(name)) for name in self.spectra}

    @property
    def observation_seconds(self) -> float:
        """The observation time in seconds."""
        return self.config.population.observation_seconds

    @property
    def segment_duration(self) -> float | None:
        """The noise segment length in seconds; ``None`` for no noise."""
        return self.config.segment_duration

    def noise(self) -> dict[str, NetworkNoise]:
        """Each network's :class:`NetworkNoise` on this background's grid.

        Derived from the config, not stored.
        """
        registry = self.config.detectors
        grid = np.asarray(self.frequencies, dtype=np.float64)
        years = self.observation_seconds / years_to_seconds(1.0)
        noise = {}
        for name in self.spectra:
            detectors = registry.build_network(name)
            psd = effective_psd(grid, detectors)
            noise[name] = NetworkNoise(
                pair_overlap=np.stack(
                    [term.overlap for term in pair_terms(grid, detectors)]
                ),
                pair_scale=np.asarray(pair_bin_scale(grid, detectors, years)),
                effective_psd=np.asarray(psd),
                scale=np.asarray(gaussian_bin_scale(jnp.asarray(psd), years, grid)),
            )
        return noise

    def write(self, path: str | Path) -> Path:
        """Write the spectra to the HDF5 file ``path``, atomically.

        Datasets ``frequencies`` and, per network,
        ``networks/<name>/pair_spectral_density``. The root attributes are
        ``metadata``, the config as JSON, ``version``, the package version,
        and ``realization``, a JSON of ``rng_key``, ``count``,
        ``total_merger_rate`` and ``hyperparameters``.

        Parameters
        ----------
        path
            File path.
        """
        with atomic_h5(path) as file:
            file.attrs["metadata"] = self.config.model_dump_json()
            file.attrs["version"] = __version__
            file.attrs["realization"] = json.dumps(
                {
                    "rng_key": list(self.rng_key),
                    "count": self.count,
                    "total_merger_rate": self.total_merger_rate,
                    "hyperparameters": self.hyperparameters,
                }
            )
            file.create_dataset("frequencies", data=self.frequencies)
            for name, spectrum in self.spectra.items():
                group = file.create_group(f"networks/{name}")
                group.create_dataset("pair_spectral_density", data=spectrum)
        return Path(path)

    @classmethod
    def open(cls, path: str | Path) -> Self:
        """Read the background at ``path``, config included.

        Parameters
        ----------
        path
            File path.

        Raises
        ------
        ValueError
            If the file's ``version`` attribute is missing or is not the
            installed astrogwb version.
        """
        with h5py.File(path, "r") as file:
            version = file.attrs.get("version")
            if version != __version__:
                raise ValueError(
                    f"{path} was written by astrogwb {version}, but {__version__} "
                    "is installed; regenerate it with astrogwb simulate"
                )
            config = SimulationConfig.model_validate_json(file.attrs["metadata"])
            realization = json.loads(file.attrs["realization"])
            frequencies = np.asarray(file["frequencies"][()], dtype=np.float64)
            stored = file["networks"]
            spectra = {
                name: np.asarray(
                    stored[name]["pair_spectral_density"][()], dtype=np.float64
                )
                for name in sorted(stored)
            }
        return cls(
            frequencies=frequencies,
            spectra=spectra,
            config=config,
            rng_key=tuple(int(v) for v in realization["rng_key"]),
            count=int(realization["count"]),
            total_merger_rate=float(realization["total_merger_rate"]),
            hyperparameters={
                name: float(value)
                for name, value in realization["hyperparameters"].items()
            },
        )


class BackgroundSimulator:
    """Generates waveforms chunk by chunk, projects onto networks and sums pair cross-powers.

    ``simulator(population)`` accumulates cross-powers over fixed-size padded
    source chunks, one fused JIT call per chunk: one compile whatever the
    realized count, and peak memory of one chunk. Build once and call often.

    A ``segment_duration`` in seconds assigns every source to a segment and
    adds sampled detector noise to the pair spectra, segment by segment (see
    the module docstring); ``None`` leaves them noiseless. ``segment_batch_size``
    is the number of segments whose strains are held at once; like
    ``chunk_size`` it changes cost, never the result.
    """

    def __init__(
        self,
        waveform: WaveformMetadata,
        registry: DetectorRegistry,
        *,
        chunk_size: int = 1024,
        validate_sources: bool = True,
        segment_duration: float | None = None,
        segment_batch_size: int = 1024,
    ) -> None:
        jax.config.update("jax_enable_x64", True)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if segment_batch_size <= 0:
            raise ValueError("segment_batch_size must be positive")
        if segment_duration is not None and segment_duration <= 0:
            raise ValueError("segment_duration must be positive")
        self._segment_duration = segment_duration
        self._segment_batch_size = segment_batch_size
        self._waveform = waveform
        self._validate_sources = validate_sources
        self._generator: WaveformGenerator = waveform_generator(waveform)
        self._frequencies = np.asarray(self._generator.frequencies, dtype=np.float64)
        self._registry = registry
        self._network_names = select_networks(registry)
        names = used_detectors(registry, self._network_names)
        rows = {pair: row for row, pair in enumerate(combinations(names, 2))}
        self._rows = {
            name: [rows[pair] for pair in registry.pairs(name)]
            for name in self._network_names
        }
        detectors = registry.build_detectors(names)
        self._tensors = np.stack([det.geometry.response_tensor for det in detectors])
        self._locations = np.stack([det.geometry.location for det in detectors])
        self._pairs = list(combinations(range(len(detectors)), 2))
        self._psds = np.stack(
            [np.asarray(det.sensitivity(self._frequencies)) for det in detectors]
        )
        self._chunk_size = chunk_size
        self._pair_sums = jax.jit(self._chunk_pair_sums)
        self._segment_sums = jax.jit(
            self._chunk_segment_sums, static_argnames="num_segments"
        )

    @property
    def waveform(self) -> WaveformMetadata:
        """The waveform this simulator generates."""
        return self._waveform

    @property
    def frequencies(self) -> NDArray[np.float64]:
        """The frequency grid in Hz, ``(F,)``."""
        return self._frequencies

    @property
    def registry(self) -> DetectorRegistry:
        """The registry used to configure this simulator."""
        return self._registry

    @property
    def segment_duration(self) -> float | None:
        """The noise segment length in seconds; ``None`` for no noise."""
        return self._segment_duration

    @property
    def chunk_size(self) -> int:
        """The number of sources evaluated per chunk."""
        return self._chunk_size

    def _chunk_strains(
        self,
        frequencies: jax.Array,
        sources: Mapping[str, jax.Array],
        valid: jax.Array,
        tensors: jax.Array,
        locations: jax.Array,
    ) -> jax.Array:
        """The ``(D, n, F)`` complex detector strains of one padded chunk.

        Padding rows have zero strain.
        """
        plus, cross = self._generator.generate_polarizations_batch(sources)
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

    def _pair_cross(self, strains: jax.Array) -> jax.Array:
        """The ``(P_union, F)`` real cross-powers summed over the sources."""
        return jnp.stack(
            [
                jnp.real(jnp.conj(strains[first]) * strains[second]).sum(axis=0)
                for first, second in self._pairs
            ]
        )

    def _chunk_pair_sums(
        self,
        frequencies: jax.Array,
        sources: Mapping[str, jax.Array],
        valid: jax.Array,
        tensors: jax.Array,
        locations: jax.Array,
    ) -> jax.Array:
        """The ``(P_union, F)`` cross-power summed over one padded chunk's sources."""
        strains = self._chunk_strains(frequencies, sources, valid, tensors, locations)
        return self._pair_cross(strains)

    def _chunk_segment_sums(
        self,
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
        strains = self._chunk_strains(frequencies, sources, valid, tensors, locations)
        coherent = jax.ops.segment_sum(
            jnp.moveaxis(strains, 1, 0), local_ids, num_segments=num_segments
        )
        return self._pair_cross(strains), coherent

    def __call__(self, population: PopulationArtifact) -> BackgroundArtifact:
        """The per-network spectra of ``population``.

        Raises
        ------
        KeyError
            If the sources carry no sky position or polarization angle.
        ValueError
            If ``validate_sources`` and a source value is outside what the
            approximant represents, or the observation is shorter than
            ``segment_duration``.
        """
        missing = {"declination", "right_ascension", "polarization_angle"} - set(
            population.source_parameters
        )
        if missing:
            raise KeyError(f"population has no extrinsic columns {sorted(missing)}")
        if self._validate_sources and population.count:
            self._generator.check_sources(population.source_parameters)
        frequencies = self._frequencies
        grid = jnp.asarray(frequencies)
        if self._segment_duration is None:
            union = self._noiseless(population, grid)
        else:
            union = self._segmented(population, grid)
        spectra = {name: union[self._rows[name]] for name in self._network_names}
        return BackgroundArtifact(
            frequencies=frequencies,
            spectra=spectra,
            config=SimulationConfig(
                population=population.metadata,
                waveform=self._waveform,
                detectors=self._registry,
                segment_duration=self._segment_duration,
            ),
            rng_key=population.rng_key,
            count=population.count,
            total_merger_rate=population.total_merger_rate,
            hyperparameters=dict(population.hyperparameters),
        )

    def _noiseless(
        self, population: PopulationArtifact, grid: jax.Array
    ) -> NDArray[np.float64]:
        """The ``(P_union, F)`` incoherent pair spectra, ``5 K / T``."""
        total = jnp.zeros((len(self._pairs), grid.shape[0]))
        for start, _ in iter_rows(population.count, self._chunk_size):
            rows, valid = pad_rows(
                population.source_parameters, start, self._chunk_size
            )
            total = total + self._pair_sums(
                grid,
                rows,
                jnp.asarray(valid),
                jnp.asarray(self._tensors),
                jnp.asarray(self._locations),
            )
        return (
            PAIR_RESPONSE_DENOMINATOR
            * np.asarray(total, dtype=np.float64)
            / population.observation_seconds
        )

    def _segmented(
        self, population: PopulationArtifact, grid: jax.Array
    ) -> NDArray[np.float64]:
        """The ``(P_union, F)`` pair spectra with per-segment sources and noise."""
        assert self._segment_duration is not None
        seconds = population.observation_seconds
        segments = int(seconds // self._segment_duration)
        if segments < 1:
            raise ValueError(
                f"segment_duration {self._segment_duration} s exceeds the "
                f"observation time {seconds} s"
            )
        effective = seconds / segments
        widths = np.asarray(bin_widths(grid), dtype=np.float64)
        if effective * widths.min() < 1.0:
            raise ValueError(
                f"segment_duration {self._segment_duration} s is shorter than "
                "1 / bin width"
            )
        wrapped = jax.random.wrap_key_data(
            jnp.asarray(population.rng_key, dtype=jnp.uint32)
        )
        labels = np.asarray(
            jax.random.randint(
                jax.random.fold_in(wrapped, SEGMENT_STREAM),
                (population.count,),
                0,
                segments,
            )
        )
        order = np.argsort(labels, kind="stable")
        labels = labels[order]
        columns = {
            name: np.asarray(values)[order]
            for name, values in population.source_parameters.items()
        }
        noise_key = jax.random.fold_in(wrapped, NOISE_STREAM)
        first, second = (jnp.asarray(index) for index in zip(*self._pairs, strict=True))
        psds = jnp.asarray(self._psds)
        tensors = jnp.asarray(self._tensors)
        locations = jnp.asarray(self._locations)
        batch = min(self._segment_batch_size, segments)
        incoherent = jnp.zeros((len(self._pairs), grid.shape[0]))
        coherent = jnp.zeros((len(self._pairs), grid.shape[0]))
        for low in range(0, segments, batch):
            lo, hi = np.searchsorted(labels, [low, low + batch])
            rows = {name: values[lo:hi] for name, values in columns.items()}
            strain = jnp.zeros((batch, self._psds.shape[0], grid.shape[0]), complex)
            for start, _ in iter_rows(int(hi - lo), self._chunk_size):
                chunk, valid = pad_rows(rows, start, self._chunk_size)
                ids = np.zeros(self._chunk_size, dtype=np.int32)
                ids[:valid] = labels[lo + start : lo + start + valid] - low
                chunk_total, chunk_strain = self._segment_sums(
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
                effective,
                segments,
            )
        ratio = 1.0 / np.sqrt(effective * widths)
        return (
            PAIR_RESPONSE_DENOMINATOR
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


def _combine(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``numerator / denominator``, zero in the bins without a usable weight."""
    usable = denominator > 0.0
    return np.where(usable, numerator / np.where(usable, denominator, 1.0), 0.0)


def combine_pair_spectra(
    pair_spectral_density: ArrayLike,
    frequencies: ArrayLike,
    registry: DetectorRegistry,
    network_name: str,
) -> NDArray[np.float64]:
    r"""The network spectrum of per-pair spectra ``(..., P, F)``, ``(..., F)``.

    :math:`\sum_{a<b} w_{ab} Y_{ab} / \sum_{a<b} w_{ab} \gamma_{ab}` with the
    optimal weights :math:`w_{ab} = \gamma_{ab} / (S_a S_b)` of
    :func:`~astrogwb.detector.pair_terms`, the pairs in the order of
    :meth:`DetectorRegistry.pairs`; its mean is :math:`S_h` for an
    isotropic, unpolarized population. Bins where no pair has a finite weight
    return zero.

    Parameters
    ----------
    pair_spectral_density
        The pair spectra :math:`Y_{ab}`, ``(..., P, F)``.
    frequencies
        The frequency grid in Hz, ``(F,)``.
    registry
        The detector registry.
    network_name
        The network of ``registry`` the pairs belong to.

    Returns
    -------
    numpy.ndarray
        The network spectrum, ``(..., F)``.
    """
    detectors = registry.build_network(network_name)
    terms = pair_terms(np.asarray(frequencies), detectors)
    weights = np.stack([term.weight for term in terms])
    denominator = np.sum([term.weight * term.overlap for term in terms], axis=0)
    numerator = np.sum(weights * np.asarray(pair_spectral_density), axis=-2)
    return _combine(numerator, denominator)
