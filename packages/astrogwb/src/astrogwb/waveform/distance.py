"""GW-luminosity-distance correction applied to a waveform catalog."""

from __future__ import annotations

import numpy as np
import xarray as xr

from astrogwb.cosmology import log_gw_em_ratio


def apply_gw_distance_to_waveforms(
    catalog: xr.Dataset,
    *,
    xi_0: float,
    xi_n: float,
) -> xr.Dataset:
    """Return a fresh catalog with polarizations rescaled for live GW propagation.

    Catalog polarizations are generated at the fiducial electromagnetic
    luminosity distance and therefore include ``1 / d_L,em^2``. Live GW
    propagation scales the effective distance by ``xi(z) = xi_0 +
    (1 - xi_0) * (1+z)^(-xi_n)``, so the corrected polarization power is
    ``power / xi(z)^2``. Applied per-sample at the amplitude level:

        plus_corrected  = plus  / xi(z)
        cross_corrected = cross / xi(z)

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
        New catalog with corrected ``polarizations``. All other data
        (``frequency``, ``source_parameters``, waveform metadata) is
        preserved as-is.
    """
    redshift = catalog.source_parameters.sel(parameter="redshift")
    log_xi = log_gw_em_ratio(redshift.values, xi_0=xi_0, xi_n=xi_n)
    inv_xi = xr.DataArray(np.exp(-log_xi), dims="sample")  # amplitude factor 1 / xi(z)

    return catalog.assign(polarizations=catalog.polarizations * inv_xi)
