"""Closed-form inspiral polarization-power generator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.waveform.analytical import inspiral_polarization_power
from astrogwb.waveform.generator.base import PolarizationPowerGenerator

__all__ = ["AnalyticInspiralGenerator"]


@dataclass(frozen=True, slots=True)
class AnalyticInspiralGenerator(PolarizationPowerGenerator):
    """Generate closed-form inspiral polarization power on the owned grid."""

    alpha: float

    def __call__(self, source_parameters: Mapping[str, ArrayLike]) -> NDArray[Any]:
        prepared_parameters = {
            name: np.asarray(values) for name, values in source_parameters.items()
        }
        return np.asarray(
            inspiral_polarization_power(
                self.frequencies,
                prepared_parameters,
                alpha=self.alpha,
            )
        ).T
