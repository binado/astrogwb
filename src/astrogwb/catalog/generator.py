"""Polarization-power generator protocol and the closed-form inspiral adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.waveform.analytical import inspiral_polarization_power
from astrogwb.waveform.metadata import FrequencyDomainWaveformMetadata

__all__ = [
    "AnalyticInspiralGenerator",
    "PolarizationPowerGenerator",
]


class PolarizationPowerGenerator(Protocol):
    """Adapter from prepared sources and an exact grid to polarization power."""

    def __call__(
        self,
        source_parameters: Mapping[str, ArrayLike],
        waveform_metadata: FrequencyDomainWaveformMetadata,
    ) -> ArrayLike: ...


@dataclass(frozen=True, slots=True)
class AnalyticInspiralGenerator:
    """Generate closed-form inspiral polarization power on the supplied grid."""

    alpha: float

    def __call__(
        self,
        source_parameters: Mapping[str, ArrayLike],
        waveform_metadata: FrequencyDomainWaveformMetadata,
    ) -> NDArray[Any]:
        prepared_parameters = {
            name: np.asarray(values) for name, values in source_parameters.items()
        }
        return np.asarray(
            inspiral_polarization_power(
                waveform_metadata.frequencies,
                prepared_parameters,
                alpha=self.alpha,
            )
        ).T
