"""The inputs that determine one simulated background, recorded with its file."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from astrogwb.detector.config import DetectorRegistry
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["SimulationConfig"]


class SimulationConfig(BaseModel):
    """Everything that determines a simulated background, but its draw.

    ``waveform`` and ``detectors`` set how events are projected onto the
    networks. ``observer_rate_hz`` is the rate, in events per second, of the
    population the event pool represents. The observation is ``num_segments``
    segments of ``segment_seconds`` each. With ``incoherent`` the spectra are
    the noiseless sums of event powers; otherwise each segment adds coherent
    strain and detector noise (see :mod:`astrogwb.simulators.background`).

    ``population``, ``seed`` and ``index`` record where the pool came from and
    the key the draw was made at. They are provenance: they do not change how
    the spectra are computed. Execution details such as chunk sizes change cost,
    never the result, and are not recorded.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    detectors: DetectorRegistry
    observer_rate_hz: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    num_segments: Annotated[int, Field(ge=1)]
    segment_seconds: Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
    incoherent: bool = False
    population: str
    seed: int
    index: Annotated[int, Field(ge=0)] = 0

    @property
    def observation_seconds(self) -> float:
        """The observed time in seconds: ``num_segments * segment_seconds``."""
        return self.num_segments * self.segment_seconds
