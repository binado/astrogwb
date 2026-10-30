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

:math:`\phi_{ab,i}` the inter-site delay phase. Detector strains are projected directly from the polarizations. Sources are
summed without cross terms between
them (no source interference), :math:`D_{ab} = \sum_i K_{ab,i}`, and normalized
by the observation time :math:`T` as

.. math::

    Y_{ab} = \frac{5\, D_{ab}}{T}.

Over an isotropic sky and uniform polarization angle
:math:`\langle K_{ab} \rangle = I \gamma_{ab} / 5`, with :math:`I` the
polarization power and :math:`\gamma_{ab}` the normalized isotropic overlap
reduction function of the native geometry. For an isotropic, unpolarized
population this gives :math:`E[Y_{ab}] = \gamma_{ab} S_h`. This is convention version
:data:`NORMALIZATION_VERSION`, recorded in every ``[background]`` table. The sum
is never divided by the realized count.

The stage is configured by a :class:`~astrogwb.waveform.WaveformMetadata` and
named networks, a :data:`DetectorRegistry`. Every pair of the union of their
detectors is projected once per chunk in a single fused kernel (waveform
evaluation plus detector projection), so a pair two networks share costs one
projection, and each network keeps its own rows, ``(P_n, F)``, with their
:func:`combine_pair_spectra`, the optimally weighted network spectrum ``(F,)``.
The combination weights come from the networks' noise curves, so the background
depends on them and they are part of its key.

Without a ``segment_duration`` the background is deterministic given its
inputs and carries no detector noise. With one, :math:`\tau`, the observation is
cut into :math:`N_\mathrm{seg} = \lfloor T/\tau \rfloor` segments of length
:math:`\tau_\mathrm{eff} = T / N_\mathrm{seg}`, and per segment :math:`s`
Gaussian noise :math:`n_a^s \sim \mathcal{CN}(0, \tau_\mathrm{eff} S_a / 2)` is
drawn in each detector and bin, together with the segment's coherent signal
strain :math:`h_a = \sqrt{\tau_\mathrm{eff}/T} \sum_i H_{a,i}`. The pair
spectrum becomes

.. math::

    Y_{ab} = \frac{5}{T} \sum_i K_{ab,i}
    + \frac{1}{\sqrt{\tau_\mathrm{eff} \Delta f_i}} \frac{1}{N_\mathrm{seg}}
    \sum_s \frac{5}{\tau_\mathrm{eff}}
    \mathrm{Re}\left[h_a^* n_b^s + n_a^{s*} h_b + n_a^{s*} n_b^s\right].

The mean stays the incoherent signal; the strain enters only through the
zero-mean cross terms. The per-bin factor rescales the averaged fluctuation to
the :math:`\Delta f_i` of the grid, so the noise-by-noise term has the standard
deviation :func:`~astrogwb.detector.pair_bin_scale` for any grid and any
:math:`\tau`. The noise stream is folded from the realization's key, so the
background stays a deterministic function of the realization. Bins where a PSD
is not finite get no noise. ``pair_spectral_density`` then holds signal plus
noise, and :meth:`BackgroundArtifact.noise` gives the matching likelihood inputs.
Detectors are static: the Earth's rotation during a signal is not modelled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

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
from astrogwb.simulators._io import atomic_h5
from astrogwb.simulators.population import PopulationArtifact
from astrogwb.utils import years_to_seconds
from astrogwb.utils.io import content_key, h5py_module
from astrogwb.waveform.metadata import WaveformMetadata

if TYPE_CHECKING:
    from astrogwb.waveform import WaveformGenerator

__all__ = [
    "NOISE_STREAM",
    "NORMALIZATION_VERSION",
    "WAVEFORM_CONVENTIONS",
    "BackgroundArtifact",
    "BackgroundSimulator",
    "NetworkNoise",
    "NetworkSpectrum",
    "background_key",
    "background_pairs",
    "background_table",
    "combine_pair_spectra",
    "select_networks",
    "waveform_key",
    "waveform_table",
]


PROJECTION_CONVENTIONS = {
    "geometry": "WGS-84 geodetic; tilted arms; metres",
    "polarization_basis": "gwmock/LAL",
    "coordinates": "right_ascension, declination; radians; compatible with gmst",
    "gmst_rad": 0.0,
    "projection": "(F_plus * plus + F_cross * cross) * exp(-2 pi i f delay)",
    "delay": "-dot(source_direction, location) / c",
    "isotropic_orf": "native WGS-84 geometry; exact invariant factorization",
}

#: The conventions a waveform manifest records, so a reader can check them.
WAVEFORM_CONVENTIONS = {
    "frame": "radiation",
    "polarization_angle": "applied_at_projection",
    "fourier": "h(f) = int h(t) exp(-2 pi i f t) dt",
    "strain_unit": "1 / Hz",
    "frequency_unit": "Hz",
}

#: :math:`\langle K_{ab} \rangle = I \gamma_{ab}` over this, the factor in
#: :math:`Y_{ab} = 5 D_{ab} / T`.
PAIR_RESPONSE_DENOMINATOR = 5.0

#: The version of the pair-spectrum convention :math:`E[Y_{ab}] = \gamma_{ab} S_h`.
NORMALIZATION_VERSION = 1

#: The stream folded into a realization's key for the detector noise.
NOISE_STREAM = 1

_NORMALIZATION = {
    "convention": "E[Y_ab] = gamma_ab * S_h",
    "pair_spectrum": "Y_ab = 5 * D_ab / T",
    "factor": PAIR_RESPONSE_DENOMINATOR,
    "version": NORMALIZATION_VERSION,
}


class NetworkSpectrum(NamedTuple):
    """One network's background: its pair spectra and their combination.

    ``pair_spectral_density`` is :math:`Y_{ab}` in :math:`\\mathrm{Hz}^{-1}`,
    ``(P_n, F)``, the pairs in the order of
    :meth:`DetectorRegistry.pairs`; ``spectral_density`` is their
    :func:`combine_pair_spectra`, ``(F,)``.
    """

    pair_spectral_density: NDArray[np.float64]
    spectral_density: NDArray[np.float64]


class NetworkNoise(NamedTuple):
    """One network's noise on the background's grid, the likelihood inputs.

    ``pair_overlap`` is the isotropic ORF :math:`\\gamma_{ab}` of each pair,
    ``(P_n, F)``; ``pair_scale`` the per-pair Gaussian scale of
    :func:`~astrogwb.detector.pair_bin_scale`, ``(P_n, F)``; ``effective_psd``
    the network's :func:`~astrogwb.detector.effective_psd`, ``(F,)``; ``scale``
    its :func:`~astrogwb.detector.gaussian_bin_scale`, ``(F,)``. The pairs are
    in the order of :meth:`DetectorRegistry.pairs`. Frequency bands and padding
    are analysis choices and are left to the reader.
    """

    pair_overlap: NDArray[np.float64]
    pair_scale: NDArray[np.float64]
    effective_psd: NDArray[np.float64]
    scale: NDArray[np.float64]


def waveform_key(metadata: WaveformMetadata, population_key: str) -> str:
    """The content hash of the waveforms of the population ``population_key``."""
    return content_key(
        {
            "population": population_key,
            "waveform": metadata.model_dump(mode="json"),
        }
    )


def waveform_table(metadata: WaveformMetadata, population_key: str) -> dict[str, Any]:
    """The ``[waveforms]`` manifest table of a population's waveforms."""
    return {
        "metadata": metadata.model_dump(mode="json"),
        "key": waveform_key(metadata, population_key),
        "conventions": {**WAVEFORM_CONVENTIONS, "source_ordering": "source_id"},
    }


def select_networks(
    registry: DetectorRegistry, names: Sequence[str] | None = None
) -> tuple[str, ...]:
    """The sorted network names ``names`` (default all) of ``registry``.

    The one place the rule that a network holds at least two detectors is
    enforced, since a network without a pair has no background.

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
    """The sorted detectors that appear in the networks ``names`` of ``registry``."""
    return tuple(sorted(registry.select(names).detectors))


def background_key(
    waveform_key: str,
    registry: DetectorRegistry,
    names: Sequence[str] | None = None,
    segment_duration: float | None = None,
) -> str:
    """The content hash of the spectra of the waveforms ``waveform_key``.

    Parameters
    ----------
    waveform_key
        The key of the polarizations the spectra are projected from.
    registry
        The detector registry.
    names
        The networks of ``registry`` the spectra cover; defaults to all.
    segment_duration
        The noise segment length in seconds; ``None`` for no noise.

    Returns
    -------
    str
        Only the membership, geometry and PSD references of the detectors those
        networks use, and the segment duration if there is noise, affect it.
    """
    identity: dict[str, Any] = {
        "waveforms": waveform_key,
        "detectors": registry.select(select_networks(registry, names)).key(),
    }
    if segment_duration is not None:
        identity["noise"] = {"segment_duration": float(segment_duration)}
    return content_key(identity)


def background_table(
    waveform_key: str,
    registry: DetectorRegistry,
    names: Sequence[str] | None = None,
    segment_duration: float | None = None,
) -> dict[str, Any]:
    """The ``[background]`` manifest table of the spectra :func:`background_key` names.

    Parameters
    ----------
    waveform_key
        The key of the polarizations the spectra are projected from.
    registry
        The detector registry.
    names
        The networks of ``registry`` the spectra cover; defaults to all.
    segment_duration
        The noise segment length in seconds; ``None`` for no noise.

    Returns
    -------
    dict
        The pairs of each network, the key, the normalization convention and
        the units, and the noise settings if there is noise.
    """
    table = {
        "metadata": {
            "pairs": {
                name: [list(pair) for pair in registry.pairs(name)]
                for name in select_networks(registry, names)
            }
        },
        "key": background_key(waveform_key, registry, names, segment_duration),
        "conventions": PROJECTION_CONVENTIONS,
        "normalization": _NORMALIZATION,
        "units": {
            "frequency": "Hz",
            "pair_spectral_density": "1 / Hz",
            "spectral_density": "1 / Hz",
        },
    }
    if segment_duration is not None:
        table["noise"] = {
            "segment_duration": float(segment_duration),
            "stream": NOISE_STREAM,
        }
    return table


def background_pairs(
    table: Mapping[str, Any],
) -> dict[str, tuple[tuple[str, str], ...]]:
    """The detector pairs of each network recorded in a ``[background]`` table.

    The pairs are in the order of the pair axis of the network's spectra.

    Raises
    ------
    ValueError
        If the table predates the recording of pairs.
    """
    if "pairs" not in table["metadata"]:
        raise ValueError("background metadata format is old; regenerate realizations")
    return {
        name: tuple((first, second) for first, second in members)
        for name, members in table["metadata"]["pairs"].items()
    }


@dataclass(frozen=True)
class BackgroundArtifact:
    """The per-network spectra of one realization, in memory or read from disk.

    ``spectra`` holds each network's :class:`NetworkSpectrum`, keyed by
    network name in sorted order, and ``pairs`` the detector pairs of its pair
    axis. ``count``, ``total_merger_rate`` (per second) and
    ``hyperparameters`` are the realization's, carried over from its
    population. ``waveform_key`` names the polarizations the spectra were
    projected from.
    """

    frequencies: NDArray[np.float64]
    spectra: dict[str, NetworkSpectrum]
    pairs: dict[str, tuple[tuple[str, str], ...]]
    observation_seconds: float
    count: int
    total_merger_rate: float
    hyperparameters: dict[str, float]
    waveform_key: str
    identity: str | None = None
    chunk_size: int | None = None
    segment_duration: float | None = None

    def key(self) -> str:
        """The recorded background identity."""
        if self.identity is None:
            raise ValueError("background identity has not been assigned")
        return self.identity

    def table(self, registry: DetectorRegistry) -> dict[str, Any]:
        """The ``[background]`` table using the supplied registry."""
        return background_table(
            self.waveform_key, registry, tuple(self.spectra), self.segment_duration
        )

    def noise(self, registry: DetectorRegistry) -> dict[str, NetworkNoise]:
        """Each network's :class:`NetworkNoise` on this background's grid.

        Derived from the registry, the grid and the observation time rather
        than stored.

        Raises
        ------
        ValueError
            If ``registry`` is not the one the background was simulated with.
        """
        names = tuple(self.spectra)
        key = background_key(self.waveform_key, registry, names, self.segment_duration)
        if key != self.key():
            raise ValueError(
                f"registry gives background {key}, not the recorded {self.key()}"
            )
        grid = np.asarray(self.frequencies, dtype=np.float64)
        years = self.observation_seconds / years_to_seconds(1.0)
        noise = {}
        for name in names:
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
        ``networks/<name>/pair_spectral_density`` and
        ``networks/<name>/spectral_density``. The pair axis is in the order of
        ``pairs``.
        """
        with atomic_h5(path) as file:
            file.create_dataset("frequencies", data=self.frequencies)
            for name, spectrum in self.spectra.items():
                group = file.create_group(f"networks/{name}")
                group.create_dataset(
                    "pair_spectral_density", data=spectrum.pair_spectral_density
                )
                group.create_dataset("spectral_density", data=spectrum.spectral_density)
        return Path(path)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        table: Mapping[str, Any],
        realization: Mapping[str, Any],
        waveform_key: str,
    ) -> Self:
        """Read the spectra at ``path`` with their manifest tables.

        ``table`` is the ``[background]`` table, ``realization`` the
        ``[realization]`` table and ``waveform_key`` the ``[waveforms]`` key.
        The recorded identity and noise segment duration are kept as is, and the file is trusted to hold
        the networks and pairs of ``table``.

        Raises
        ------
        ValueError
            If the table uses another normalization convention or predates
            the recording of pairs.
        """
        version = table["normalization"].get("version")
        if version != NORMALIZATION_VERSION:
            raise ValueError(
                f"{path} uses pair-spectrum normalization version {version}, "
                f"not {NORMALIZATION_VERSION}"
            )
        pairs = background_pairs(table)
        h5py = h5py_module()
        with h5py.File(path, "r") as file:
            frequencies = np.asarray(file["frequencies"][()], dtype=np.float64)
            stored = file["networks"]
            spectra = {
                name: NetworkSpectrum(
                    np.asarray(
                        stored[name]["pair_spectral_density"][()], dtype=np.float64
                    ),
                    np.asarray(stored[name]["spectral_density"][()], dtype=np.float64),
                )
                for name in pairs
            }
        return cls(
            frequencies=frequencies,
            spectra=spectra,
            pairs=pairs,
            observation_seconds=float(realization["observation_seconds"]),
            count=int(realization["count"]),
            total_merger_rate=float(realization["total_merger_rate"]),
            hyperparameters={
                name: float(value)
                for name, value in realization["hyperparameters"].items()
            },
            waveform_key=waveform_key,
            identity=str(table["key"]),
            segment_duration=table.get("noise", {}).get("segment_duration"),
        )


class BackgroundSimulator:
    """Generates waveforms chunk by chunk, projects onto networks and sums pair cross-powers.

    ``simulator(population)`` iterates over fixed-size padded source chunks of
    ``population``, evaluates the radiation-frame polarizations and accumulates
    their cross-powers over the detector union's pairs in one fused JIT call
    per chunk, so a pair shared by networks is projected once. Each network then
    takes its rows and their :func:`combine_pair_spectra`, using the registry's
    resolved geometry and PSD references. One chunk shape is one compile,
    whatever the realized count, and peak memory is one chunk of polarizations.
    The observation time comes from the population. Build once and call often.

    A ``segment_duration`` in seconds adds sampled detector noise to the pair
    spectra, segment by segment (see the module docstring); ``None`` leaves
    them noiseless.
    """

    def __init__(
        self,
        waveform: WaveformMetadata,
        registry: DetectorRegistry,
        *,
        chunk_size: int = 1024,
        validate_sources: bool = True,
        segment_duration: float | None = None,
    ) -> None:
        jax.config.update("jax_enable_x64", True)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if segment_duration is not None and segment_duration <= 0:
            raise ValueError("segment_duration must be positive")
        self._segment_duration = segment_duration
        self._waveform = waveform
        self._validate_sources = validate_sources
        self._generator: WaveformGenerator = waveform.build()
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

    def _chunk_pair_sums(
        self,
        frequencies: jax.Array,
        sources: Mapping[str, jax.Array],
        valid: jax.Array,
        tensors: jax.Array,
        locations: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """The cross-power and strain summed over one padded chunk's sources.

        Returns the ``(P_union, F)`` real cross-powers and the ``(D, F)``
        complex strain of the detector union.
        """
        plus, cross = self._generator.generate_polarizations_batch(sources)
        real = (jnp.arange(plus.shape[0]) < valid)[:, None]
        plus = jnp.where(real, plus, 0.0)
        cross = jnp.where(real, cross, 0.0)
        strains = jax.vmap(
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
        cross = jnp.stack(
            [
                jnp.real(jnp.conj(strains[first]) * strains[second]).sum(axis=0)
                for first, second in self._pairs
            ]
        )
        return cross, strains.sum(axis=1)

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
        total = jnp.zeros((len(self._pairs), frequencies.shape[0]))
        strain = jnp.zeros((self._psds.shape[0], frequencies.shape[0]), complex)
        grid = jnp.asarray(frequencies)
        for start, _ in iter_rows(population.count, self._chunk_size):
            rows, valid = pad_rows(
                population.source_parameters, start, self._chunk_size
            )
            chunk_total, chunk_strain = self._pair_sums(
                grid,
                rows,
                jnp.asarray(valid),
                jnp.asarray(self._tensors),
                jnp.asarray(self._locations),
            )
            total = total + chunk_total
            strain = strain + chunk_strain
        union = (
            PAIR_RESPONSE_DENOMINATOR
            * np.asarray(total, dtype=np.float64)
            / population.observation_seconds
        )
        if self._segment_duration is not None:
            union = union + self._noise(population, grid, strain)
        spectra = {}
        for name in self._network_names:
            pairs = union[self._rows[name]]
            spectra[name] = NetworkSpectrum(
                pairs,
                combine_pair_spectra(pairs, frequencies, self._registry, name),
            )
        wf_key = waveform_key(self._waveform, population.key())
        return BackgroundArtifact(
            frequencies=frequencies,
            spectra=spectra,
            pairs={
                name: tuple(self._registry.pairs(name)) for name in self._network_names
            },
            observation_seconds=population.observation_seconds,
            count=population.count,
            total_merger_rate=population.total_merger_rate,
            hyperparameters=dict(population.hyperparameters),
            waveform_key=wf_key,
            identity=background_key(
                wf_key, self._registry, self._network_names, self._segment_duration
            ),
            chunk_size=self._chunk_size,
            segment_duration=self._segment_duration,
        )

    def _noise(
        self, population: PopulationArtifact, grid: jax.Array, strain: jax.Array
    ) -> NDArray[np.float64]:
        """The ``(P_union, F)`` sampled noise terms of the pair spectra."""
        assert self._segment_duration is not None
        segments = int(population.observation_seconds // self._segment_duration)
        if segments < 1:
            raise ValueError(
                f"segment_duration {self._segment_duration} s exceeds the "
                f"observation time {population.observation_seconds} s"
            )
        key = jax.random.fold_in(
            jax.random.wrap_key_data(jnp.asarray(population.rng_key, dtype=jnp.uint32)),
            NOISE_STREAM,
        )
        first, second = (jnp.asarray(index) for index in zip(*self._pairs, strict=True))
        return np.asarray(
            _sample_noise(
                key,
                strain,
                jnp.asarray(self._psds),
                bin_widths(grid),
                first,
                second,
                population.observation_seconds,
                num_segments=segments,
            ),
            dtype=np.float64,
        )


@partial(jax.jit, static_argnames="num_segments")
def _sample_noise(
    key: jax.Array,
    signal_strain: jax.Array,
    psds: jax.Array,
    widths: jax.Array,
    first: jax.Array,
    second: jax.Array,
    observation_seconds: jax.Array | float,
    *,
    num_segments: int,
) -> jax.Array:
    """The noise terms of the pair spectra, ``(P, F)``; see the module docstring.

    ``signal_strain`` is the ``(D, F)`` complex strain summed over sources,
    ``psds`` the ``(D, F)`` noise curves, ``widths`` the ``(F,)`` bin widths
    and ``first``, ``second`` the detector indices of each pair. One compile
    per ``num_segments``.
    """
    segment = observation_seconds / num_segments
    finite = jnp.isfinite(psds)
    amplitude = jnp.sqrt(segment * jnp.where(finite, psds, 0.0) / 4.0)
    coherent = jnp.sqrt(segment / observation_seconds) * signal_strain

    def accumulate(total: jax.Array, segment_key: jax.Array) -> tuple[jax.Array, None]:
        real_key, imag_key = jax.random.split(segment_key)
        noise = amplitude * jax.lax.complex(
            jax.random.normal(real_key, psds.shape),
            jax.random.normal(imag_key, psds.shape),
        )
        cross = (
            jnp.conj(coherent[first]) * noise[second]
            + jnp.conj(noise[first]) * coherent[second]
            + jnp.conj(noise[first]) * noise[second]
        )
        return total + jnp.real(cross), None

    zero = jnp.zeros((first.shape[0], psds.shape[1]))
    total, _ = jax.lax.scan(accumulate, zero, jax.random.split(key, num_segments))
    return (
        PAIR_RESPONSE_DENOMINATOR
        * total
        / (num_segments * segment * jnp.sqrt(segment * widths))
    )


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
    ------
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
