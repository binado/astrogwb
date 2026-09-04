"""Network effective PSD and per-bin Gaussian noise scale.

This module owns the detector/network noise model seen by a stochastic
background search: the inverse-variance network ``effective_psd`` and the
per-bin Gaussian scale ``σ = S_eff / √(2 T Δf)``. Overlap-reduction geometry
stays in ``overlap``; GWB signal spectra stay in ``astrogwb.gwb``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from gwmock_signal.stochastic.overlap import detector_names
from numpy.typing import ArrayLike, NDArray

from astrogwb.utils import years_to_seconds

from ._types import DetectorSpec
from .overlap import overlap_reduction_function
from .sensitivity import Sensitivity, evaluate_psd

_NETWORK_SENSITIVITY_FACTOR = 0.16


def effective_psd(
    frequencies: ArrayLike,
    detectors: Sequence[DetectorSpec],
    sensitivities: Mapping[str, Sensitivity],
) -> NDArray[np.float64]:
    """Network effective PSD from an inverse-variance cross-correlation sum.

    ``detectors`` are matched to ``sensitivities`` by their public name
    (gwmock ``detector_names``): plain str codes match directly, while
    preset ``CustomDetector`` objects match on their ``name`` (e.g.
    ``ET1_SARD``). The contraction and ``_NETWORK_SENSITIVITY_FACTOR``
    are unchanged from the pre-refactor implementation.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    det_list = list(detectors)
    if len(det_list) < 2:
        return np.full(frequencies.shape, np.inf, dtype=float)

    names = detector_names(det_list)
    psds = [
        evaluate_psd(sensitivities[name].psd_reference, frequencies) for name in names
    ]

    inverse_variance = np.zeros(frequencies.shape, dtype=float)
    for i, det1 in enumerate(det_list):
        for j in range(i + 1, len(det_list)):
            gamma = overlap_reduction_function(frequencies, det1, det_list[j])
            with np.errstate(invalid="ignore", divide="ignore"):
                contribution = gamma**2 / (psds[i] * psds[j])
            inverse_variance += np.where(np.isfinite(contribution), contribution, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 1.0 / (_NETWORK_SENSITIVITY_FACTOR * np.sqrt(inverse_variance))
    return np.where(inverse_variance > 0.0, result, np.inf)


def gaussian_bin_scale(
    effective_psd: jax.Array,
    observation_time: float,
    df: float | jax.Array,
) -> jax.Array:
    """Per-bin Gaussian noise scale for a stochastic background search.

    Parameters
    ----------
    observation_time:
        Observation time in years. Converted to seconds internally.
    df:
        Frequency-bin width in Hz.
    """
    observation_time_sec = years_to_seconds(observation_time)
    return effective_psd / jnp.sqrt(2.0 * observation_time_sec * df)
