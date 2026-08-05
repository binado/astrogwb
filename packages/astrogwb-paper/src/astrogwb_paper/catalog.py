"""Waveform-catalog preprocessing shared by the paper MCMC entrypoints.

The D1/S11 GW-distance correction is applied once, at the fixed fiducial
modified-propagation parameters, *before* polarization power is computed. With a
GW-referenced catalog the importance weights stay pure EM-distance (no live
xi factor), which matches the convention documented in ``PYTHON_REFACTOR.md``.
"""

from __future__ import annotations

from astrogwb.waveform import apply_gw_distance_to_waveforms
from pluscross import WaveformCatalog


def apply_gw_distance_at_fiducial(
    catalog: WaveformCatalog,
    *,
    xi_0: float,
    xi_n: float,
) -> WaveformCatalog:
    """Rescale catalog polarizations to live-GW distances at the fiducial xi.

    Returns a fresh catalog with ``plus``/``cross`` divided by
    ``xi(z) = xi_0 + (1 - xi_0)(1+z)^(-xi_n)`` per sample, so polarization
    power carries the ``1 / xi^2`` correction for modified GW propagation.
    With the GR fiducial ``xi_0 = 1`` this is the identity transform.

    The correction is applied once at the fixed fiducial point, before
    polarization power is computed; importance weights therefore carry only
    the EM-distance ratio.
    """
    return apply_gw_distance_to_waveforms(catalog, xi_0=xi_0, xi_n=xi_n)
