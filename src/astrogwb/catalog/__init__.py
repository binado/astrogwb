"""Array-native catalogs with their own population record and I/O.

Two artifacts, one metadata record. A
:class:`PolarizationPowerCatalog` persists per-source waveform power; a
:class:`SpectralDensityCatalog` persists the forward model's contraction of it.
Both carry the :class:`~astrogwb.metadata.PopulationMetadata` that produced
them, and both are read and written through the same HDF5 layer.

:func:`simulate` is the cache in front of any generator: hand it a metadata
record and a generator -- :class:`SpectrumGenerator` for spectral-density
draws -- and it serves ``<cache_dir>/<key>.h5`` or generates and saves it.
"""

from astrogwb.catalog.cache import (
    Artifact,
    Generator,
    Keyed,
    artifact_path,
    catalog_path,
    check_catalog_answers,
    generate,
    load_or_generate,
    save_atomically,
    simulate,
)
from astrogwb.catalog.polarization_power import REDSHIFT_SITE, PolarizationPowerCatalog
from astrogwb.catalog.spectra import SpectrumGenerator
from astrogwb.catalog.spectral_density import SpectralDensityCatalog

__all__ = [
    "REDSHIFT_SITE",
    "Artifact",
    "Generator",
    "Keyed",
    "PolarizationPowerCatalog",
    "SpectralDensityCatalog",
    "SpectrumGenerator",
    "artifact_path",
    "catalog_path",
    "check_catalog_answers",
    "generate",
    "load_or_generate",
    "save_atomically",
    "simulate",
]
