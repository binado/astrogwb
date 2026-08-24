"""Waveform-catalog IO and consumption helpers.

Owns the ``waveform_catalog`` on-disk format (see :mod:`astrogwb.waveform.catalog`)
and reduces loaded catalogs to the derived quantities the inference stack needs.
"""

from astrogwb.waveform.catalog import (
    WaveformCatalog,
    load_catalog,
    make_catalog,
    open_catalog,
    save_catalog,
    validate_catalog,
)
from astrogwb.waveform.distance import apply_gw_distance_to_power
from astrogwb.waveform.polarization_power import polarization_power

__all__ = [
    "WaveformCatalog",
    "apply_gw_distance_to_power",
    "load_catalog",
    "make_catalog",
    "open_catalog",
    "polarization_power",
    "save_catalog",
    "validate_catalog",
]
