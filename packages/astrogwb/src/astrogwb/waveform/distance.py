"""GW-luminosity-distance correction applied to a waveform catalog."""

from __future__ import annotations

import numpy as np
from pluscross import WaveformCatalog

from astrogwb.cosmology import log_gw_em_ratio


def apply_gw_distance_to_waveforms(
    catalog: WaveformCatalog,
    *,
    xi_0: float,
    xi_n: float,
) -> WaveformCatalog:
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
        Loaded ``pluscross.WaveformCatalog``. ``source_parameters`` must
        include ``redshift`` and ``luminosity_distance`` (both kept untouched).
    xi_0, xi_n:
        Live modified-propagation parameters.

    Returns
    -------
    WaveformCatalog
        New catalog with corrected ``plus``/``cross``. All other fields
        (``frequencies``, ``source_parameters``, waveform metadata) are
        preserved as-is.
    """
    redshift = catalog.source_parameters["redshift"]
    log_xi = log_gw_em_ratio(redshift, xi_0=xi_0, xi_n=xi_n)
    inv_xi = np.exp(-log_xi)  # (nsamples,), amplitude factor 1 / xi(z)

    return WaveformCatalog(
        frequencies=catalog.frequencies,
        plus=catalog.plus * inv_xi[:, None],
        cross=catalog.cross * inv_xi[:, None],
        source_parameters=dict(catalog.source_parameters),
        approximant=catalog.approximant,
        minimum_frequency=catalog.minimum_frequency,
        maximum_frequency=catalog.maximum_frequency,
        reference_frequency=catalog.reference_frequency,
        sampling_frequency=catalog.sampling_frequency,
    )
