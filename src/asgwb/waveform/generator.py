from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from .grid import FrequencyGrid

if TYPE_CHECKING:
    from bilby.gw.waveform_generator import WaveformGenerator as BilbyWaveformGenerator
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

    @classmethod
    def from_sampling(
        cls,
        approximant: str,
        duration: float,
        sampling_frequency: float,
        reference_frequency: float,
        source_type: SourceType = "BNS",
        minimum_frequency: float = 10.0,
        maximum_frequency: float | None = None,
    ) -> WaveformGenerator:
        grid = FrequencyGrid(
            duration=duration,
            sampling_frequency=sampling_frequency,
            reference_frequency=reference_frequency,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
        return cls(approximant=approximant, grid=grid, source_type=source_type)

    def frequency_domain_polarizations(
        self, parameters: dict[str, float]
    ) -> WaveformPolarizations:
        return self._backend.frequency_domain_polarizations(parameters)

    def as_bilby_waveform_generator(self) -> BilbyWaveformGenerator:
        from ._bilby import BilbyWaveformBackend

        if not isinstance(self._backend, BilbyWaveformBackend):
            backend = BilbyWaveformBackend(
                self.approximant, self.grid, self.source_type
            )
            return backend.waveform_generator
        return self._backend.waveform_generator
