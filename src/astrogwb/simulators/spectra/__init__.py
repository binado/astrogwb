"""Forward-model spectral-density draws, drawn from a :class:`SpectraMetadata`."""

from astrogwb.simulators.spectra.catalog import SpectralDensityCatalog
from astrogwb.simulators.spectra.forward import (
    PackedPowerSum,
    normalize_spectra,
    validate_source_model,
)
from astrogwb.simulators.spectra.metadata import Hyperparameter, SpectraMetadata
from astrogwb.simulators.spectra.poisson_counts import poisson_counts_forward_model
from astrogwb.simulators.spectra.simulator import (
    SpectraSimulator,
    get_simulator,
    spectra,
)

__all__ = [
    "Hyperparameter",
    "PackedPowerSum",
    "SpectraMetadata",
    "SpectraSimulator",
    "SpectralDensityCatalog",
    "get_simulator",
    "normalize_spectra",
    "poisson_counts_forward_model",
    "spectra",
    "validate_source_model",
]
