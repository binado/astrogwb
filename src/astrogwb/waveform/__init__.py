"""Waveform-catalog consumption helpers.

Catalog IO lives in the external ``waveform_catalog`` package (see the
waveform-catalog repo); this package only reduces loaded catalogs to the
derived quantities the inference stack needs.
"""

from astrogwb.waveform.polarization_power import polarization_power

__all__ = [
    "polarization_power",
]
