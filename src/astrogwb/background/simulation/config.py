"""The inputs that determine one simulated background, recorded with its file."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrogwb.detector import DetectorConfig, DetectorNetworkConfig
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["SimulationConfig"]


def _network_detectors(
    networks: Sequence[DetectorNetworkConfig],
) -> dict[str, DetectorConfig]:
    """Resolve the shared definitions and validate that the networks can run together."""
    if not networks:
        raise ValueError("a background needs at least one network")
    names: set[str] = set()
    detectors: dict[str, DetectorConfig] = {}
    for network in networks:
        if network.name in names:
            raise ValueError(f"duplicate network name {network.name!r}")
        names.add(network.name)
        if len(network.detectors) < 2:
            raise ValueError(
                f"network {network.name!r} must contain at least two detectors"
            )
        for name, detector in network.detectors.items():
            previous = detectors.setdefault(name, detector)
            if previous != detector:
                raise ValueError(f"conflicting definitions for detector {name!r}")
    return detectors


class SimulationConfig(BaseModel):
    """Everything that determines a simulated background, but its draw.

    ``waveform`` and ``networks`` set how events are projected onto the
    networks. ``observer_rate_hz`` is the rate, in events per second, of the
    population the event pool represents. The observation is ``num_segments``
    segments of ``segment_seconds`` each. With ``incoherent`` the spectra are
    the noiseless sums of event powers; otherwise each segment adds coherent
    strain and detector noise (see :mod:`astrogwb.background.simulation.draw`).

    ``population``, ``seed`` and ``index`` record where the pool came from and
    the key the draw was made at. They are provenance: they do not change how
    the spectra are computed. Execution details such as chunk sizes change cost,
    never the result, and are not recorded.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformMetadata
    networks: tuple[DetectorNetworkConfig, ...]
    observer_rate_hz: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    num_segments: Annotated[int, Field(ge=1)]
    segment_seconds: Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
    incoherent: bool = False
    population: str
    seed: int
    index: Annotated[int, Field(ge=0)] = 0

    @field_validator("networks")
    @classmethod
    def _validate_networks(
        cls, networks: tuple[DetectorNetworkConfig, ...]
    ) -> tuple[DetectorNetworkConfig, ...]:
        _network_detectors(networks)
        return networks

    @property
    def observation_seconds(self) -> float:
        """The observed time in seconds: ``num_segments * segment_seconds``."""
        return self.num_segments * self.segment_seconds
