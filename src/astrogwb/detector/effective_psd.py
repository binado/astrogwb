r"""Network effective PSD and per-bin Gaussian noise scale.

The inverse-variance network ``effective_psd`` and the per-bin Gaussian scale
:math:`\sigma_i = S_{\mathrm{eff},i} / \sqrt{2 T \Delta f_i}`. Observation times are
in years.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.frequency import bin_widths
from astrogwb.utils import years_to_seconds

from .config import Network, map_over_pairs
from .overlap import network_overlap_reduction_function

# Normalization of the overlap reduction function: the paper's
# Gamma_ab = (2/5) gamma_ab, with gamma_ab the normalized ORF returned by
# ``isotropic_overlap_reduction_function`` (one for co-located aligned L-shaped detectors).
_OVERLAP_REDUCTION_FUNCTION_NORM = 0.4


def _noise_product(frequencies: NDArray[np.float64], network: Network) -> NDArray:
    """Product of the two detectors' noise PSDs of every pair, ``(P, F)``."""
    products = map_over_pairs(
        lambda first, second: (
            first.sensitivity(frequencies) * second.sensitivity(frequencies)
        ),
        network,
    )
    return np.stack(tuple(products.values()))


def pair_weights(frequencies: ArrayLike, network: Network) -> NDArray[np.float64]:
    r"""Optimal cross-correlation weights of every pair, ``(P, F)``.

    :math:`w_{ab} = \gamma_{ab} / (S_{n,a} S_{n,b})`, with pairs in the order of
    :func:`~astrogwb.detector.network_pairs` and zero where the weight is not
    finite (out of band).

    Parameters
    ----------
    frequencies
        Frequency grid in Hz.
    network
        Detectors keyed by name; at least two.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    overlap = np.asarray(network_overlap_reduction_function(frequencies, network))
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = overlap / _noise_product(frequencies, network)
    return np.where(np.isfinite(weight), weight, 0.0)


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
        Detectors keyed by name. With fewer than two the PSD is infinite.

    Returns
    -------
    numpy.ndarray
        :math:`S_{\mathrm{eff}}` in Hz^-1, aligned with ``frequencies``.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if len(network) < 2:
        return np.full(frequencies.shape, np.inf, dtype=float)

    overlap = np.asarray(network_overlap_reduction_function(frequencies, network))
    inverse_variance = np.sum(pair_weights(frequencies, network) * overlap, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 / (_OVERLAP_REDUCTION_FUNCTION_NORM * np.sqrt(inverse_variance))
    return np.where(inverse_variance > 0.0, result, np.inf)


def network_cross_power_std(
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
    effective_psd
        Network effective PSD, aligned with ``frequencies``.
    observation_time
        Observation time in years.
    frequencies
        The frequency grid in Hz. Bin widths come from
        :func:`astrogwb.frequency.bin_widths`, so pass the catalog's *full*
        axis and mask afterwards rather than a selected sub-grid.
    """
    seconds = years_to_seconds(observation_time)
    return effective_psd / jnp.sqrt(2.0 * seconds * bin_widths(frequencies))


def pair_cross_power_std(
    frequencies: ArrayLike,
    network: Network,
    observation_time: float,
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
        Detectors keyed by name; at least two.
    observation_time
        Observation time in years.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    seconds = years_to_seconds(observation_time)
    modes = np.sqrt(2.0 * seconds * np.asarray(bin_widths(frequencies)))
    with np.errstate(invalid="ignore"):
        scale = np.sqrt(_noise_product(frequencies, network)) / (
            _OVERLAP_REDUCTION_FUNCTION_NORM * modes
        )
    return jnp.asarray(np.where(np.isnan(scale), np.inf, scale))
