r"""Network effective PSD and per-bin Gaussian noise scale.

This module owns the detector/network noise model seen by a stochastic
background search: the inverse-variance network ``effective_psd`` and the
per-bin Gaussian scale :math:`\sigma_i = S_{\mathrm{eff},i} / \sqrt{2 T \Delta f_i}`,
plus its grid-independent plotting counterpart
:math:`\sigma_{\ln f} = S_{\mathrm{eff}} / \sqrt{2 T f}`. Overlap-reduction
geometry stays in ``overlap``; GWB signal spectra stay in ``astrogwb.gwb``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gwmock_signal.stochastic.overlap import detector_names
from numpy.typing import ArrayLike, NDArray

from astrogwb.frequency import bin_widths
from astrogwb.utils import years_to_seconds

from ._types import DetectorSpec
from .overlap import isotropic_overlap_reduction_function
from .sensitivity import Sensitivity

# Normalization of the overlap reduction function: the paper's
# Gamma_ab = (2/5) gamma_ab, with gamma_ab the normalized ORF returned by
# ``isotropic_overlap_reduction_function`` (one for co-located aligned L-shaped detectors).
_OVERLAP_REDUCTION_FUNCTION_NORM = 0.4


class PairTerms(NamedTuple):
    r"""One detector pair of an optimally weighted cross-correlation network.

    Attributes
    ----------
    first, second:
        Indices of the two detectors in the network, ``first < second``.
    weight:
        :math:`w_{ab} = \gamma_{ab} / (S_{n,a} S_{n,b})`, zero where it is not
        finite (out of band), ``(F,)``.
    overlap:
        The isotropic ORF :math:`\gamma_{ab}`, ``(F,)``.
    """

    first: int
    second: int
    weight: NDArray[np.float64]
    overlap: NDArray[np.float64]


def pair_terms(
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
    sensitivities: Mapping[str, Sensitivity],
) -> list[PairTerms]:
    r"""The optimal weight and isotropic ORF of every pair :math:`a < b`.

    The weights of the optimal cross-correlation estimator, shared by
    :func:`effective_psd` (:math:`S_{\mathrm{eff}}^{-2} \propto \sum w_{ab}
    \gamma_{ab}`) and :func:`~astrogwb.detector.response.network_response`.
    ``detectors`` are matched to ``sensitivities`` by their public name (gwmock
    ``detector_names``): plain str codes match directly, while a
    ``CustomDetector`` matches on its ``name``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    det_list = list(detectors)
    psds = [sensitivities[name](frequencies) for name in detector_names(det_list)]
    pairs = []
    for i, det1 in enumerate(det_list):
        for j in range(i + 1, len(det_list)):
            gamma = isotropic_overlap_reduction_function(frequencies, det1, det_list[j])
            with np.errstate(invalid="ignore", divide="ignore"):
                weight = gamma / (psds[i] * psds[j])
            pairs.append(
                PairTerms(i, j, np.where(np.isfinite(weight), weight, 0.0), gamma)
            )
    return pairs


def effective_psd(
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
    sensitivities: Mapping[str, Sensitivity],
) -> NDArray[np.float64]:
    r"""Network effective PSD from an inverse-variance cross-correlation sum.

    .. math::

        S_{\mathrm{eff}}^{-2} = \sum_{a<b}
        \frac{\Gamma_{ab}^2}{S_{n,a} S_{n,b}}, \qquad
        \Gamma_{ab} = \tfrac{2}{5}\, \gamma_{ab},

    with :math:`\gamma_{ab}` the normalized overlap reduction function, summed
    over :func:`pair_terms` as :math:`\sum_{a<b} w_{ab} \gamma_{ab}`.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if len(list(detectors)) < 2:
        return np.full(frequencies.shape, np.inf, dtype=float)

    inverse_variance = np.zeros(frequencies.shape, dtype=float)
    for pair in pair_terms(frequencies, detectors, sensitivities):
        inverse_variance += pair.weight * pair.overlap

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 / (_OVERLAP_REDUCTION_FUNCTION_NORM * np.sqrt(inverse_variance))
    return np.where(inverse_variance > 0.0, result, np.inf)


def gaussian_bin_scale(
    effective_psd: jax.Array,
    observation_time: float,
    frequencies: ArrayLike,
) -> jax.Array:
    r"""Per-bin Gaussian noise scale for a stochastic background search.

    .. math::

        \sigma_i = \frac{S_{\mathrm{eff},i}}{\sqrt{2 T \Delta f_i}}

    A bin of width :math:`\Delta f_i` averages :math:`T \Delta f_i` Fourier
    modes, which is why the scale narrows on a coarser bin.

    Parameters
    ----------
    effective_psd:
        Network effective PSD, aligned with ``frequencies``.
    observation_time:
        Observation time in years. Converted to seconds internally.
    frequencies:
        The frequency grid in Hz. Bin widths come from
        :func:`astrogwb.frequency.bin_widths`, so pass the catalog's *full*
        axis and mask afterwards rather than a selected sub-grid.
    """
    observation_time_sec = years_to_seconds(observation_time)
    return effective_psd / jnp.sqrt(
        2.0 * observation_time_sec * bin_widths(frequencies)
    )


def pair_bin_scale(
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
    sensitivities: Mapping[str, Sensitivity],
    observation_time: float,
) -> jax.Array:
    r"""Per-bin Gaussian noise scale of every pair's cross-power, ``(P, F)``.

    .. math::

        \sigma_{ab,i} = \frac{\sqrt{S_{n,a} S_{n,b}}}{\tfrac{2}{5}
            \sqrt{2 T \Delta f_i}},

    the scale of a pair's cross-power whose mean is :math:`\gamma_{ab} S_h`
    (the per-pair spectra of
    :class:`~astrogwb.simulators.spectra.NetworkSpectralDensitySimulator`),
    pairs in the order of :func:`pair_terms`. It is not divided by
    :math:`\gamma_{ab}`, so it stays finite at the ORF's zeros, and the pairs
    combine to the network: :math:`\sum_{a<b} \gamma_{ab,i}^2 /
    \sigma_{ab,i}^2 = 1/\sigma_i^2` with :math:`\sigma_i` the
    :func:`gaussian_bin_scale` of :func:`effective_psd`. Infinite where a noise
    curve is.

    Parameters
    ----------
    frequencies:
        The frequency grid in Hz; bin widths as for :func:`gaussian_bin_scale`.
    detectors, sensitivities:
        As for :func:`effective_psd`.
    observation_time:
        Observation time in years.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    det_list = list(detectors)
    psds = [
        np.asarray(sensitivities[name](frequencies), dtype=float)
        for name in detector_names(det_list)
    ]
    modes = np.sqrt(
        2.0 * years_to_seconds(observation_time) * np.asarray(bin_widths(frequencies))
    )
    with np.errstate(invalid="ignore"):
        scale = np.stack(
            [
                np.sqrt(psds[first] * psds[second])
                for first in range(len(det_list))
                for second in range(first + 1, len(det_list))
            ]
        ) / (_OVERLAP_REDUCTION_FUNCTION_NORM * modes)
    return jnp.asarray(np.where(np.isnan(scale), np.inf, scale))


def log_frequency_noise_scale(
    effective_psd: jax.Array,
    frequencies: jax.Array,
    observation_time: float,
) -> jax.Array:
    r"""Grid-independent sensitivity curve: the noise scale per e-fold of frequency.

    .. math::

        \sigma_{\ln f}(f) = \frac{S_{\mathrm{eff}}(f)}{\sqrt{2 T f}}

    This is :func:`gaussian_bin_scale` with the bin width replaced by
    :math:`f`, so it does not move when the frequency grid is refined. It
    is a *presentation* scale for sensitivity plots, not the likelihood's
    noise scale: :math:`(S_h/\sigma_{\ln f})^2` is
    :math:`d\mathrm{SNR}^2/d\ln f`
    (:func:`astrogwb.gwb.spectral_snr_squared_per_log_frequency`), so the
    gap between a spectrum and this curve shows where the SNR accrues.

    Parameters
    ----------
    effective_psd:
        Network effective PSD in Hz^-1.
    frequencies:
        Frequencies in Hz, aligned with ``effective_psd``.
    observation_time:
        Observation time in years. Converted to seconds internally.
    """
    observation_time_sec = years_to_seconds(observation_time)
    return effective_psd / jnp.sqrt(2.0 * observation_time_sec * frequencies)
