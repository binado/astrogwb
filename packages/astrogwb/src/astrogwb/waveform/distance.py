"""GW-luminosity-distance correction applied to a waveform catalog."""

from __future__ import annotations

import numpy as np
import xarray as xr

from astrogwb.cosmology import log_gw_em_ratio


def apply_gw_distance_to_power(
    catalog: xr.Dataset,
    *,
    xi_0: float,
    xi_n: float,
) -> xr.Dataset:
    """Return a fresh catalog with polarization power rescaled for live GW propagation.

    Catalog polarization power is generated at the fiducial electromagnetic
    luminosity distance and therefore includes ``1 / d_L,em^4`` (power scales
    as amplitude squared). Live GW propagation scales the effective distance
    by ``xi(z) = xi_0 + (1 - xi_0) * (1+z)^(-xi_n)``, so the corrected power is

        power_corrected = power / xi(z)^2

    Parameters
    ----------
    catalog:
        Loaded waveform catalog. ``source_parameters`` must include
        ``redshift`` and ``luminosity_distance`` (both kept untouched).
    xi_0, xi_n:
        Live modified-propagation parameters.

    Returns
    -------
    xr.Dataset
        New catalog with corrected ``polarization_power``. All other data
        (``frequency``, ``source_parameters``, waveform metadata) is
        preserved as-is.
    """
    redshift = catalog.source_parameters.sel(parameter="redshift")
    log_xi = log_gw_em_ratio(redshift.values, xi_0=xi_0, xi_n=xi_n)
    inv_xi_sq = xr.DataArray(
        np.exp(-2.0 * log_xi), dims="sample"
    )  # power factor 1/xi(z)^2

    return catalog.assign(polarization_power=catalog.polarization_power * inv_xi_sq)
