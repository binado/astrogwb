"""Array-native catalogs with their own population record and I/O.

Two artifacts, one metadata record. A
:class:`PolarizationPowerCatalog` persists per-source waveform power; a
:class:`SpectralDensityCatalog` persists the forward model's contraction of it.
Both carry the :class:`~astrogwb.metadata.PopulationMetadata` that produced
them, and both are read and written through the same HDF5 layer.
"""

from astrogwb.catalog.cache import (
    catalog_path,
    check_catalog_answers,
    generate,
    load_or_generate,
    save_atomically,
)
from astrogwb.catalog.polarization_power import REDSHIFT_SITE, PolarizationPowerCatalog
from astrogwb.catalog.spectral_density import SpectralDensityCatalog

__all__ = [
    "REDSHIFT_SITE",
    "PolarizationPowerCatalog",
    "SpectralDensityCatalog",
    "catalog_path",
    "check_catalog_answers",
    "generate",
    "load_or_generate",
    "save_atomically",
]
