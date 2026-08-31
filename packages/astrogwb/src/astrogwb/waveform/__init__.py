"""Waveform-catalog IO, consumption helpers, and a closed-form inspiral model.

Owns the ``waveform_catalog`` on-disk format (see :mod:`astrogwb.waveform.catalog`)
and reduces loaded catalogs to the derived quantities the inference stack needs.
:mod:`astrogwb.waveform.analytical` supplies the quadrupolar, inspiral-only
polarization power in closed form, as an alternative to a numerically
generated bank.
"""

from astrogwb.waveform.analytical import (
    chirp_mass,
    inclination_factor,
    inspiral_polarization_power,
    termination_frequency,
)
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
    "chirp_mass",
    "inclination_factor",
    "inspiral_polarization_power",
    "load_catalog",
    "make_catalog",
    "open_catalog",
    "polarization_power",
    "save_catalog",
    "termination_frequency",
    "validate_catalog",
]
