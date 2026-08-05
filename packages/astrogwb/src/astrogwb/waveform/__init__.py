"""Waveform-catalog consumption helpers.

Catalog IO lives in the external ``pluscross`` package; this package only
reduces loaded catalogs to the derived quantities the inference stack needs.
"""

from astrogwb.waveform.distance import apply_gw_distance_to_waveforms
from astrogwb.waveform.polarization_power import polarization_power

__all__ = [
    "apply_gw_distance_to_waveforms",
    "polarization_power",
]
