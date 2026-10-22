r"""Spectral densities as a cross-correlation network records them.

:class:`~astrogwb.simulators.spectra.BackgroundSpectralDensitySimulator` sums
each source's sky-averaged power :math:`I_i = |h_+|^2 + |h_\times|^2`. A
detector network does not record that: every source reaches each pair through
its own antenna patterns and inter-site delay. This simulator keeps that
response. Each source's Stokes parameters are contracted, pair by pair, with
:func:`~astrogwb.detector.stokes_cross_power` at an isotropic sky position and
uniform polarization angle, giving per-pair sums
:math:`D_{ab} = \sum_i K_{ab, i}`. The pairs are then combined as the optimal
cross-correlation estimator does,

.. math::

    S_h^{\mathrm{data}}(f) = k\, \frac{5 \sum_{a<b} w_{ab}\, D_{ab}}
        {\sum_{a<b} w_{ab}\, \gamma_{ab}}, \qquad
    w_{ab} = \frac{\gamma_{ab}}{S_a S_b},

with :math:`\gamma_{ab}` the isotropic ORF, the weights of
:func:`~astrogwb.detector.pair_terms` and :math:`k` the normalization of
:func:`~astrogwb.simulators.spectra.forward.normalize_spectra`. Averaged over
orientations :math:`\langle K_{ab} \rangle = I\, \gamma_{ab}/5`, so the mean is
the sky-averaged spectrum, and the scatter about it is the per-source response
that :class:`~astrogwb.inference.likelihood.Network` models as a Hadamard
covariance. It is the noise-free estimator: detector noise is not added.

The inclination enters once, through the population's own :math:`\iota_i` in
the Stokes parameters. The extrinsic angles -- direction, polarization angle
and coalescence phase -- are drawn here from a key folded off the draw's, so the
same key gives the same sources as the sky-averaged simulator. The coalescence
phase replaces the population's pinned zero; only higher-mode waveforms
depend on it. Detectors are static, as for the per-source ORF: the Earth's
rotation during a long signal is not modelled.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.detector import Sensitivity, pair_terms
from astrogwb.simulators.population.simulator import PopulationData
from astrogwb.simulators.spectra.forward import ChunkedNetworkSum, normalize_spectra
from astrogwb.simulators.spectra.metadata import BackgroundSpectralDensityMetadata
from astrogwb.simulators.spectra.simulator import (
    BackgroundSpectralDensityData,
    draw_components,
)

__all__ = [
    "JointNetworkSpectralDensitySimulator",
    "NetworkSpectralDensityMetadata",
    "NetworkSpectralDensitySimulator",
    "draw_extrinsic_angles",
]

#: The isotropic mean of :math:`K_{ab}` is :math:`I \gamma_{ab}` over this.
_PAIR_RESPONSE_DENOMINATOR = 5.0


class NetworkSpectralDensityMetadata(BackgroundSpectralDensityMetadata):
    """The record a network spectral-density artifact is generated from and cached under.

    A :class:`~astrogwb.simulators.spectra.BackgroundSpectralDensityMetadata`
    plus the network that records it: ``detectors``, two or more site codes of
    :func:`~astrogwb.detector.resolve_detector`, and ``psd_references``, the
    :class:`~astrogwb.detector.Sensitivity` reference of each, keyed by site
    code. The pair weights depend on both, so both are part of :meth:`key`.
    """

    detectors: list[str]
    psd_references: dict[str, str]

    def sensitivities(self) -> dict[str, Sensitivity]:
        """The noise curve of every detector, keyed by site code."""
        return {name: Sensitivity(self.psd_references[name]) for name in self.detectors}


def draw_extrinsic_angles(key: jax.Array, count: int) -> dict[str, jax.Array]:
    """Isotropic extrinsic angles of ``count`` sources, as source columns.

    ``polar_cosine`` uniform on :math:`[-1, 1]` and ``azimuth`` on
    :math:`[0, 2\\pi)` (an isotropic direction), ``polarization`` on
    :math:`[0, \\pi)` and ``coa_phase`` on :math:`[0, 2\\pi)`, each ``(count,)``.
    """
    keys = jax.random.split(key, 4)
    shape = (count,)
    return {
        "polar_cosine": jax.random.uniform(keys[0], shape, minval=-1.0, maxval=1.0),
        "azimuth": jax.random.uniform(keys[1], shape, maxval=2.0 * jnp.pi),
        "polarization": jax.random.uniform(keys[2], shape, maxval=jnp.pi),
        "coa_phase": jax.random.uniform(keys[3], shape, maxval=2.0 * jnp.pi),
    }


class NetworkSpectralDensitySimulator:
    """Draws network spectra from :class:`NetworkSpectralDensityMetadata`.

    The network counterpart of
    :class:`~astrogwb.simulators.spectra.BackgroundSpectralDensitySimulator`,
    with the same call, settings and
    :class:`~astrogwb.simulators.spectra.BackgroundSpectralDensityData` output,
    so :func:`~astrogwb.simulators.spectra.stack_spectra` and the HDF5 layer
    apply unchanged; ``spectral_density`` is the network spectrum of the module
    docstring. The waveform must implement
    :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate_stokes_batch`.
    Build once and call often: the pair weights and the compiled reduction are
    reused.
    """

    def __init__(
        self,
        metadata: NetworkSpectralDensityMetadata,
        *,
        chunk_size: int = 128,
        source_chunk_size: int | None = None,
        validate_sources: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._metadata = metadata
        self._population, self._generator = draw_components(
            metadata,
            source_chunk_size=chunk_size
            if source_chunk_size is None
            else source_chunk_size,
            validate_sources=validate_sources,
        )
        self._reducer = ChunkedNetworkSum(
            self._generator, metadata.detectors, chunk_size=chunk_size
        )
        self._weights, self._denominator = _pair_weights(
            np.asarray(self._generator.frequencies), metadata
        )

    @property
    def metadata(self) -> NetworkSpectralDensityMetadata:
        """The record this simulator draws."""
        return self._metadata

    def combine(self, pair_sums: ArrayLike) -> NDArray[np.float64]:
        """The network spectrum of per-pair cross-powers ``(P, ..., F)``, ``(..., F)``.

        Before normalization. The pairs are in the order of
        :attr:`~astrogwb.simulators.spectra.forward.ChunkedNetworkSum.pairs`, and
        any middle axes (one source per entry, say) are kept. Bins where no pair
        has a finite weight return zero, as
        :func:`~astrogwb.detector.network_response` does.
        """
        return _combine_pairs(self._weights, self._denominator, pair_sums)

    def reduce(
        self, population: PopulationData, angle_key: jax.Array
    ) -> BackgroundSpectralDensityData:
        """The network spectrum of ``population``, its extrinsic angles drawn from ``angle_key``."""
        count = int(population["count"])
        angles = draw_extrinsic_angles(angle_key, count)
        pair_sums = self._reducer({**population["source_parameters"], **angles})
        return _network_data(
            self.combine(np.asarray(pair_sums)),
            population,
            self._metadata,
            self._population.observation_seconds,
            np.asarray(self._generator.frequencies),
        )

    def __call__(self, key: jax.Array) -> BackgroundSpectralDensityData:
        """The network spectrum of the draw at ``key``.

        The population is drawn with ``key`` itself and the extrinsic angles
        with ``jax.random.fold_in(key, 1)``.
        """
        return self.reduce(self._population(key), jax.random.fold_in(key, 1))


class JointNetworkSpectralDensitySimulator:
    """Network spectra of several networks from one pass over each draw.

    ``simulator(key)[n]`` equals
    ``NetworkSpectralDensitySimulator(metadata[n])(key)`` up to the order of
    floating-point sums: the population and the extrinsic angles are drawn
    once, from the same keys, and every network's spectrum is combined from
    the per-pair cross-powers of the union of the networks' detectors. A pair
    depends only on its two detectors, so it is computed once however many
    networks share it, and the waveforms -- the expensive half -- once per
    draw rather than once per network. Every network then sees the same
    sources at the same sky positions.

    The ``metadata`` must differ only in ``detectors`` and
    ``psd_references``, and a site code shared by two networks must name the
    same detector; this is trusted, not checked. Settings are those of
    :class:`NetworkSpectralDensitySimulator`; peak memory is
    ``(P, F, chunk_size)`` for the ``P`` distinct pairs.
    """

    def __init__(
        self,
        metadata: Sequence[NetworkSpectralDensityMetadata],
        *,
        chunk_size: int = 128,
        source_chunk_size: int | None = None,
        validate_sources: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._metadata = list(metadata)
        self._population, self._generator = draw_components(
            self._metadata[0],
            source_chunk_size=chunk_size
            if source_chunk_size is None
            else source_chunk_size,
            validate_sources=validate_sources,
        )
        detectors = list(
            dict.fromkeys(
                name for record in self._metadata for name in record.detectors
            )
        )
        pairs: list[tuple[int, int]] = []
        #: Each network's rows of the joint pair sums, in its own a < b order.
        self._rows: list[list[int]] = []
        for record in self._metadata:
            rows = []
            for first in range(len(record.detectors)):
                for second in range(first + 1, len(record.detectors)):
                    pair = (
                        detectors.index(record.detectors[first]),
                        detectors.index(record.detectors[second]),
                    )
                    if pair not in pairs:
                        pairs.append(pair)
                    rows.append(pairs.index(pair))
            self._rows.append(rows)
        self._reducer = ChunkedNetworkSum(
            self._generator, detectors, chunk_size=chunk_size, pairs=pairs
        )
        frequencies = np.asarray(self._generator.frequencies)
        self._weights = [
            _pair_weights(frequencies, record) for record in self._metadata
        ]

    @property
    def metadata(self) -> list[NetworkSpectralDensityMetadata]:
        """The records this simulator draws, in output order."""
        return list(self._metadata)

    def __call__(self, key: jax.Array) -> list[BackgroundSpectralDensityData]:
        """The spectrum of every network for the draw at ``key``, in ``metadata`` order."""
        population = self._population(key)
        count = int(population["count"])
        angles = draw_extrinsic_angles(jax.random.fold_in(key, 1), count)
        pair_sums = np.asarray(
            self._reducer({**population["source_parameters"], **angles})
        )
        frequencies = np.asarray(self._generator.frequencies)
        return [
            _network_data(
                _combine_pairs(weights, denominator, pair_sums[rows]),
                population,
                record,
                self._population.observation_seconds,
                frequencies,
            )
            for record, rows, (weights, denominator) in zip(
                self._metadata, self._rows, self._weights, strict=True
            )
        ]


def _pair_weights(
    frequencies: NDArray[np.float64], metadata: NetworkSpectralDensityMetadata
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The pair weights ``(P, F)`` and their ORF-weighted sum ``(F,)`` of a network."""
    terms = pair_terms(frequencies, metadata.detectors, metadata.sensitivities())
    weights = np.stack([term.weight for term in terms])
    denominator = np.sum([term.weight * term.overlap for term in terms], axis=0)
    return weights, denominator


def _combine_pairs(
    weights: NDArray[np.float64],
    denominator: NDArray[np.float64],
    pair_sums: ArrayLike,
) -> NDArray[np.float64]:
    """:meth:`NetworkSpectralDensitySimulator.combine` with explicit weights."""
    pair_sums = np.asarray(pair_sums)
    weights = weights.reshape(weights.shape[0], *([1] * (pair_sums.ndim - 2)), -1)
    numerator = _PAIR_RESPONSE_DENOMINATOR * np.sum(weights * pair_sums, axis=0)
    usable = denominator > 0.0
    return np.where(usable, numerator / np.where(usable, denominator, 1.0), 0.0)


def _network_data(
    combined: NDArray[np.float64],
    population: PopulationData,
    metadata: NetworkSpectralDensityMetadata,
    observation_seconds: float,
    frequencies: NDArray[np.float64],
) -> BackgroundSpectralDensityData:
    """One draw's network spectrum, normalized, as a one-row data record."""
    rate = np.asarray(population["total_merger_rate"])
    density = normalize_spectra(
        combined,
        count=metadata.count,
        total_merger_rate=rate,
        observation_seconds=observation_seconds,
        num_events=metadata.num_events,
    )
    return {
        "frequencies": frequencies,
        "spectral_density": density[np.newaxis, :],
        "n_events": np.asarray([int(population["count"])], dtype=np.int64),
        "total_merger_rate": rate[np.newaxis],
        "hyperparameters": {
            name: np.asarray(value)[np.newaxis]
            for name, value in population["hyperparameters"].items()
        },
    }
