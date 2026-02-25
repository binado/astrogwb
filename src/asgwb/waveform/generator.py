from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from .grid import FrequencyGrid
from .polarizations import WaveformPolarizations

SourceType = Literal["BBH", "BHNS", "BNS"]


@runtime_checkable
class WaveformBackend(Protocol):
    def frequency_domain_polarizations(
        self, parameters: dict[str, float]
    ) -> WaveformPolarizations: ...


@dataclass
class WaveformGenerator:
    """High-level waveform generator that delegates to a backend."""

    approximant: str
    grid: FrequencyGrid
    source_type: SourceType = "BNS"
    _backend: WaveformBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from ._bilby import BilbyWaveformBackend

        self._backend = BilbyWaveformBackend(
            self.approximant, self.grid, self.source_type
        )

    def frequency_domain_polarizations(
        self, parameters: dict[str, float]
    ) -> WaveformPolarizations:
        return self._backend.frequency_domain_polarizations(parameters)
