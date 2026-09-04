"""GW-luminosity-distance correction for polarization power."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.cosmology import log_gw_em_ratio


def apply_gw_distance_to_power(
    polarization_power: ArrayLike,
    redshift: ArrayLike,
    *,
    xi_0: float,
    xi_n: float,
) -> NDArray[np.floating]:
    """Return fresh polarization power rescaled for live GW propagation.

    Catalog polarization power is generated at the fiducial electromagnetic
    luminosity distance and therefore includes ``1 / d_L,em^4`` (power scales
    as amplitude squared). Live GW propagation scales the effective distance
    by ``xi(z) = xi_0 + (1 - xi_0) * (1+z)^(-xi_n)``, so the corrected power is

        power_corrected = power / xi(z)^2

    Parameters
    ----------
    polarization_power:
        Real frequency-first array with shape ``(frequency, sample)``.
    redshift:
        One redshift per sample.
    xi_0, xi_n:
        Live modified-propagation parameters.

    Returns
    -------
    numpy.ndarray
        A new array containing the corrected polarization power.
    """
    power = np.asarray(polarization_power)
    redshifts = np.asarray(redshift)
    if power.ndim != 2:
        raise ValueError("polarization_power must be a two-dimensional array")
    if redshifts.ndim != 1:
        raise ValueError("redshift must be a one-dimensional array")
    if power.shape[1] != redshifts.shape[0]:
        raise ValueError(
            "redshift length must match the polarization_power sample axis"
        )

    log_xi = np.asarray(log_gw_em_ratio(redshifts, xi_0=xi_0, xi_n=xi_n))
    return np.asarray(power * np.exp(-2.0 * log_xi)[None, :])
