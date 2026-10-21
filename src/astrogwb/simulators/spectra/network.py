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
        terms = pair_terms(
            np.asarray(self._generator.frequencies),
            metadata.detectors,
            metadata.sensitivities(),
        )
        self._weights = np.stack([term.weight for term in terms])
        self._denominator = np.sum(
            [term.weight * term.overlap for term in terms], axis=0
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
        pair_sums = np.asarray(pair_sums)
        weights = self._weights.reshape(
            self._weights.shape[0], *([1] * (pair_sums.ndim - 2)), -1
        )
        numerator = _PAIR_RESPONSE_DENOMINATOR * np.sum(weights * pair_sums, axis=0)
        usable = self._denominator > 0.0
        return np.where(
            usable, numerator / np.where(usable, self._denominator, 1.0), 0.0
        )

    def reduce(
        self, population: PopulationData, angle_key: jax.Array
    ) -> BackgroundSpectralDensityData:
        """The network spectrum of ``population``, its extrinsic angles drawn from ``angle_key``."""
        sources = dict(population["source_parameters"])
        count = int(population["count"])
        rate = np.asarray(population["total_merger_rate"])
        angles = draw_extrinsic_angles(angle_key, count)
        pair_sums = self._reducer({**sources, **angles})
        density = normalize_spectra(
            self.combine(np.asarray(pair_sums)),
            count=self._metadata.count,
            total_merger_rate=rate,
            observation_seconds=self._population.observation_seconds,
            num_events=self._metadata.num_events,
        )
        return {
            "frequencies": np.asarray(self._generator.frequencies),
            "spectral_density": density[np.newaxis, :],
            "n_events": np.asarray([count], dtype=np.int64),
            "total_merger_rate": rate[np.newaxis],
            "hyperparameters": {
                name: np.asarray(value)[np.newaxis]
                for name, value in population["hyperparameters"].items()
            },
        }

    def __call__(self, key: jax.Array) -> BackgroundSpectralDensityData:
        """The network spectrum of the draw at ``key``.

        The population is drawn with ``key`` itself and the extrinsic angles
        with ``jax.random.fold_in(key, 1)``.
        """
        return self.reduce(self._population(key), jax.random.fold_in(key, 1))
