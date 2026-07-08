"""Waveform-catalog consumption helpers.

Catalog IO lives in the external ``pluscross`` package; this package only
reduces loaded catalogs to the derived quantities the inference stack needs.
"""

from astrogwb.waveform.polarization_power import polarization_power

__all__ = [
    "polarization_power",
]
