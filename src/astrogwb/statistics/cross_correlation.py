r"""The optimally weighted cross-correlation statistic of a detector network.

Each pair :math:`(a, b)` measures a cross-power spectrum :math:`Y_{ab}` with
mean :math:`\gamma_{ab} S_h` for an isotropic, unpolarized background.
:func:`combine_pair_spectra` combines a network's pairs into one spectrum with
mean :math:`S_h`, weighting each pair by its inverse noise variance.
"""

from __future__ import annotations

import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from astrogwb.detector import DetectorNetworkConfig, pair_terms

__all__ = ["combine_pair_spectra"]


def _combine(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``numerator / denominator``, zero in the bins without a usable weight."""
    usable = denominator > 0.0
    return np.where(usable, numerator / np.where(usable, denominator, 1.0), 0.0)


def combine_pair_spectra(
    pair_spectral_density: ArrayLike,
    frequencies: ArrayLike,
    network: DetectorNetworkConfig,
) -> NDArray[np.float64]:
    r"""The network spectrum of per-pair spectra ``(..., P, F)``, ``(..., F)``.

    :math:`\sum_{a<b} w_{ab} Y_{ab} / \sum_{a<b} w_{ab} \gamma_{ab}` with the
    optimal weights :math:`w_{ab} = \gamma_{ab} / (S_a S_b)` of
    :func:`~astrogwb.detector.pair_terms`, the pairs in the order of
    :meth:`DetectorNetworkConfig.pairs`; its mean is :math:`S_h` for an
    isotropic, unpolarized population. Bins where no pair has a finite weight
    return zero.

    Parameters
    ----------
    pair_spectral_density
        The pair spectra :math:`Y_{ab}`, ``(..., P, F)``.
    frequencies
        The frequency grid in Hz, ``(F,)``.
    network
        The resolved network the pairs belong to.

    Returns
    -------
    numpy.ndarray
        The network spectrum, ``(..., F)``.
    """
    detectors = network.build_detectors()
    terms = pair_terms(np.asarray(frequencies), detectors)
    weights = np.stack([term.weight for term in terms])
    denominator = np.sum([term.weight * term.overlap for term in terms], axis=0)
    numerator = np.sum(weights * np.asarray(pair_spectral_density), axis=-2)
    return _combine(numerator, denominator)
