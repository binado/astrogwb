"""Array-native catalogs with their own population record and I/O.

Two artifacts, one metadata record. A
:class:`PolarizationPowerCatalog` persists per-source waveform power; a
:class:`SpectralDensityCatalog` persists the forward model's contraction of it.
Both carry the :class:`~astrogwb.populations.PopulationRecord` that produced
them, and both are read and written through the same HDF5 layer.
"""

from astrogwb.catalog.polarization_power import REDSHIFT_SITE, PolarizationPowerCatalog
from astrogwb.catalog.spectral_density import SpectralDensityCatalog

__all__ = ["REDSHIFT_SITE", "PolarizationPowerCatalog", "SpectralDensityCatalog"]
