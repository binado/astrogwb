"""Array-native catalogs with their own population record and I/O.

Two artifacts, each with its own metadata record. A
:class:`PolarizationPowerCatalog` persists per-source waveform power and
carries a :class:`~astrogwb.metadata.CatalogMetadata`; a
:class:`SpectralDensityCatalog` persists the forward model's contraction of it
and carries a :class:`~astrogwb.metadata.SpectraMetadata`. Both records share
the :class:`~astrogwb.metadata.PopulationMetadata` that produced the draw, and
both artifacts are read and written through the same HDF5 layer.

:func:`simulate` is the cache in front of any generator: hand it a metadata
record and a generator -- :class:`CatalogGenerator` for catalogs,
:class:`SpectrumGenerator` for spectral-density draws -- and it serves
``<cache_dir>/<key>.h5`` or generates and saves it.
"""

from astrogwb.catalog.cache import (
    Artifact,
    Generator,
    Keyed,
    artifact_path,
    check_metadata,
    save_atomically,
    simulate,
)
from astrogwb.catalog.generator import CatalogGenerator
from astrogwb.catalog.polarization_power import REDSHIFT_SITE, PolarizationPowerCatalog
from astrogwb.catalog.spectra import SpectrumGenerator
from astrogwb.catalog.spectral_density import SpectralDensityCatalog

__all__ = [
    "REDSHIFT_SITE",
    "Artifact",
    "CatalogGenerator",
    "Generator",
    "Keyed",
    "PolarizationPowerCatalog",
    "SpectralDensityCatalog",
    "SpectrumGenerator",
    "artifact_path",
    "check_metadata",
    "save_atomically",
    "simulate",
]
