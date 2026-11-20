r"""Network effective PSD and per-bin Gaussian noise scale.

The inverse-variance network ``effective_psd``, the per-bin Gaussian scale
:math:`\sigma_i = S_{\mathrm{eff},i} / \sqrt{2 T \Delta f_i}`, and its
grid-independent plotting counterpart
:math:`\sigma_{\ln f} = S_{\mathrm{eff}} / \sqrt{2 T f}`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.frequency import bin_widths

from .config import Network, map_over_pairs, network_pairs
from .overlap import isotropic_overlap_reduction_function

# Normalization of the overlap reduction function: the paper's
# Gamma_ab = (2/5) gamma_ab, with gamma_ab the normalized ORF returned by
# ``isotropic_overlap_reduction_function`` (one for co-located aligned L-shaped detectors).
_OVERLAP_REDUCTION_FUNCTION_NORM = 0.4


class _PairTerms(NamedTuple):
    r"""One detector pair of an optimally weighted cross-correlation network.

    Attributes
    ----------
    weight
        :math:`w_{ab} = \gamma_{ab} / (S_{n,a} S_{n,b})`, zero where it is not
        finite (out of band), ``(F,)``.
    overlap
        The isotropic ORF :math:`\gamma_{ab}`, ``(F,)``.
    """

    weight: NDArray[np.float64]
    overlap: NDArray[np.float64]


def _pair_terms(frequencies: ArrayLike, network: Network) -> list[_PairTerms]:
    """Return optimal weights and isotropic overlaps in canonical pair order."""
    frequencies = np.asarray(frequencies, dtype=float)
    psds = {
        name: detector.sensitivity(frequencies) for name, detector in network.items()
    }
    overlaps = map_over_pairs(
        lambda first, second: isotropic_overlap_reduction_function(
            frequencies, first, second
        ),
        network,
    )
    terms = []
    for (first, second), overlap in overlaps.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            weight = overlap / (psds[first] * psds[second])
        terms.append(_PairTerms(np.where(np.isfinite(weight), weight, 0.0), overlap))
    return terms


def effective_psd(
    frequencies: ArrayLike,
    network: Network,
) -> NDArray[np.float64]:
    r"""Network effective PSD from an inverse-variance cross-correlation sum.

    .. math::

        S_{\mathrm{eff}}^{-2} = \sum_{a<b}
        \frac{\Gamma_{ab}^2}{S_{n,a} S_{n,b}}, \qquad
        \Gamma_{ab} = \tfrac{2}{5}\, \gamma_{ab},

    with :math:`\gamma_{ab}` the normalized overlap reduction function, summed
    over :func:`~astrogwb.detector.network_pairs` as :math:`\sum_{a<b} w_{ab} \gamma_{ab}`.

    Parameters
    ----------
    frequencies
        Frequency grid in Hz.
    network
        Detectors in the network.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if len(network) < 2:
        return np.full(frequencies.shape, np.inf, dtype=float)

    inverse_variance = np.zeros(frequencies.shape, dtype=float)
    for pair in _pair_terms(frequencies, network):
        inverse_variance += pair.weight * pair.overlap

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 / (_OVERLAP_REDUCTION_FUNCTION_NORM * np.sqrt(inverse_variance))
    return np.where(inverse_variance > 0.0, result, np.inf)


def network_cross_power_std(
    effective_psd: jax.Array,
    observation_seconds: float,
    frequencies: ArrayLike,
) -> jax.Array:
    r"""Per-bin Gaussian noise scale for a stochastic background search.

    .. math::

        \sigma_i = \frac{S_{\mathrm{eff},i}}{\sqrt{2 T \Delta f_i}}

    A bin of width :math:`\Delta f_i` averages :math:`T \Delta f_i` Fourier
    modes, which is why the scale narrows on a coarser bin.

    Parameters
    ----------
    effective_psd
        Network effective PSD, aligned with ``frequencies``.
    observation_seconds
        Observation time in seconds.
    frequencies
        The frequency grid in Hz. Bin widths come from
        :func:`astrogwb.frequency.bin_widths`, so pass the catalog's *full*
        axis and mask afterwards rather than a selected sub-grid.
    """
    return effective_psd / jnp.sqrt(2.0 * observation_seconds * bin_widths(frequencies))


def pair_cross_power_std(
    frequencies: ArrayLike,
    network: Network,
    observation_seconds: float,
) -> jax.Array:
    r"""Per-bin Gaussian noise scale of every pair's cross-power, ``(P, F)``.

    .. math::

        \sigma_{ab,i} = \frac{\sqrt{S_{n,a} S_{n,b}}}{\tfrac{2}{5}
            \sqrt{2 T \Delta f_i}},

    the scale of a pair's cross-power whose mean is :math:`\gamma_{ab} S_h`
    (the pair spectra of :class:`~astrogwb.background.simulation.BackgroundDraw`,
    as stored by :meth:`~astrogwb.background.simulation.BackgroundDraw.write`),
    pairs in the order of :func:`~astrogwb.detector.network_pairs`. It is not divided by
    :math:`\gamma_{ab}`, so it stays finite at the ORF's zeros, and the pairs
    combine to the network: :math:`\sum_{a<b} \gamma_{ab,i}^2 /
    \sigma_{ab,i}^2 = 1/\sigma_i^2` with :math:`\sigma_i` the
    :func:`network_cross_power_std` of :func:`effective_psd`. Infinite where a noise
    curve is.

    Parameters
    ----------
    frequencies
        The frequency grid in Hz; bin widths as for :func:`network_cross_power_std`.
    network
        As for :func:`effective_psd`.
    observation_seconds
        Observation time in seconds.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    psds = {
        name: detector.sensitivity(frequencies) for name, detector in network.items()
    }
    modes = np.sqrt(2.0 * observation_seconds * np.asarray(bin_widths(frequencies)))
    with np.errstate(invalid="ignore"):
        scale = np.stack(
            [
                np.sqrt(psds[first] * psds[second])
                for first, second in network_pairs(network)
            ]
        ) / (_OVERLAP_REDUCTION_FUNCTION_NORM * modes)
    return jnp.asarray(np.where(np.isnan(scale), np.inf, scale))


def _log_frequency_noise_scale(
    effective_psd: jax.Array,
    frequencies: jax.Array,
    observation_seconds: float,
) -> jax.Array:
    r"""Grid-independent sensitivity curve: the noise scale per e-fold of frequency.

    .. math::

        \sigma_{\ln f}(f) = \frac{S_{\mathrm{eff}}(f)}{\sqrt{2 T f}}

    :func:`network_cross_power_std` with the bin width replaced by :math:`f`: a
    presentation scale for sensitivity plots, not the likelihood's.
    :math:`(S_h/\sigma_{\ln f})^2` is :math:`d\mathrm{SNR}^2/d\ln f`
    (:func:`astrogwb.background.spectral_snr_squared_per_log_frequency`).

    Parameters
    ----------
    effective_psd
        Network effective PSD in Hz^-1.
    frequencies
        Frequencies in Hz, aligned with ``effective_psd``.
    observation_seconds
        Observation time in seconds.
    """
    return effective_psd / jnp.sqrt(2.0 * observation_seconds * frequencies)
