from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid


def redshift_pdf(
    z: NDArray,
    cosmology,
    source_frame_distribution: NDArray,
    time_delay_fn: Callable | None,
    minimum_time_delay: float = 0.02,  # Gyr
    z_min: float | None = None,
    z_max: float | None = None,
    normalize: bool = True,
) -> NDArray:
    """Compute the redshift PDF given a source-frame merger rate distribution.

    Optionally convolves the merger rate with a time-delay distribution before
    projecting into detector-frame coordinates via the comoving volume element.
    """
    pdf = source_frame_distribution
    # Normalize source_frame distribution to 1 at z = 0 so it can be multiplied
    # by the local merger rate afterwards
    pdf /= pdf[0]
    if callable(time_delay_fn):
        # Convolve merger rate density with time delay distribution
        lookback_time = cosmology.lookback_time(z).value  # Gyr
        # In astropy, lookback time is an integration from 0 to z,
        # and therefore is an increasing function of z.
        # By defining time_delay[i, j] = t_j - t_i, we want to integrate over
        # its positive values, therefore over the j axis (=-1 in numpy)
        time_delay = lookback_time[np.newaxis, :] - lookback_time[:, np.newaxis]
        # time_delay_fn should return a normalized pdf with signature (i,j) -> (i,j)
        time_delay_pdf = time_delay_fn(time_delay, minimum_time_delay)
        dt_dz = cosmology.hubble_time.value * cosmology.lookback_time_integrand(z)
        joint_pdf = time_delay_pdf * dt_dz * pdf
        pdf = trapezoid(joint_pdf, x=z)

    dvc_dz = cosmology.differential_comoving_volume(z).value
    pdf *= 4.0 * np.pi * dvc_dz / (1.0 + z)
    if z_min is not None:
        pdf *= z >= z_min
    if z_max is not None:
        pdf *= z <= z_max
    return pdf / trapezoid(pdf, x=z) if normalize else pdf


def madau_dickinson_source_frame_distribution(
    z: NDArray, kappa: float, gamma: float, z_peak: float
) -> NDArray:
    """Madau-Dickinson star formation rate as a source-frame merger rate proxy."""
    prob = (1 + z) ** gamma / (1 + ((1 + z) / (1 + z_peak)) ** kappa)
    # Normalise to 1 at z = 0
    prob *= 1 + (1 + z_peak) ** (-kappa)
    return prob


def power_law_source_frame_distribution(z: NDArray, lamb: float) -> NDArray:
    """Power-law source-frame merger rate: (1 + z)^lamb."""
    return (1.0 + z) ** lamb


def inverse_time_delay_pdf(time_delay: NDArray, minimum_time_delay: float) -> NDArray:
    """Normalized inverse time-delay distribution, truncated at minimum_time_delay."""
    max_time_delay = np.max(time_delay, axis=1)
    integration_domain = max_time_delay > minimum_time_delay
    norm = np.ones_like(max_time_delay)
    norm[integration_domain] = np.log(
        max_time_delay[integration_domain] / minimum_time_delay
    )
    causal_slice = time_delay > minimum_time_delay
    res = np.zeros_like(time_delay)
    res[causal_slice] = 1 / time_delay[causal_slice]
    return res / norm[:, np.newaxis]


AVAILABLE_TIME_DELAY_MODELS: dict[str, Callable] = {
    "inverse_time_delay": inverse_time_delay_pdf,
}
