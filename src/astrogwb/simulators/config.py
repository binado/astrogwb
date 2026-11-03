"""The configuration that determines a simulation, one model for the whole run."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from astrogwb.detector.config import DetectorRegistry
from astrogwb.simulators.population import RealizationMetadata
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["SimulationConfig"]


class SimulationConfig(BaseModel):
    """Everything that determines a simulated background, but its draw.

    ``population`` sets the sources, ``waveform`` their polarizations and
    ``detectors`` the networks they are projected onto. A ``segment_duration``
    in seconds sets the population arrival-time bounds and adds per-segment
    detector noise (see
    :mod:`astrogwb.simulators.background`); ``None`` leaves the spectra
    noiseless. The draw, a seed and an index, is not configuration: it is the
    key the configuration is simulated at. Execution details such as chunk
    sizes change cost, never the result, and are not recorded either.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    population: RealizationMetadata
    waveform: WaveformMetadata
    detectors: DetectorRegistry
    segment_duration: Annotated[float, Field(gt=0.0)] | None = None
