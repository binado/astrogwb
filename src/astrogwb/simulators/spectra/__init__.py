"""Forward-model spectral-density draws, drawn from a :class:`SpectraMetadata`."""

from astrogwb.simulators.spectra.catalog import SpectralDensityCatalog
from astrogwb.simulators.spectra.forward import (
    ChunkedPowerSum,
    normalize_spectra,
    validate_source_model,
)
from astrogwb.simulators.spectra.metadata import Hyperparameter, SpectraMetadata
from astrogwb.simulators.spectra.poisson_counts import poisson_counts_forward_model
from astrogwb.simulators.spectra.simulator import (
    SpectraData,
    SpectraSimulator,
    Spectrum,
    stack_spectra,
)

__all__ = [
    "ChunkedPowerSum",
    "Hyperparameter",
    "SpectraData",
    "SpectraMetadata",
    "SpectraSimulator",
    "SpectralDensityCatalog",
    "Spectrum",
    "normalize_spectra",
    "poisson_counts_forward_model",
    "stack_spectra",
    "validate_source_model",
]
