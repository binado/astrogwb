"""The inputs that determine one simulated background, recorded with its file."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrogwb.detector import Network, network_pairs
from astrogwb.waveform.metadata import WaveformConfig

__all__ = ["SimulationConfig"]


class SimulationConfig(BaseModel):
    """Simulation settings and provenance stored with a background draw.

    ``waveform`` and ``network`` set how events are projected. ``poisson_rate``
    is the rate, in events per second, of the population the pool represents.
    The observation is ``num_segments`` segments of ``segment_duration`` each.
    Each segment sums coherent strain and adds detector noise when
    ``simulate_detector_noise`` is set.

    ``population``, ``population_root`` and ``seed`` record the pool location
    and the draw's seed.
    The root is ignored for CSV files. The waveform and segment batch sizes are
    execution settings and omitted from the file.
    ``network`` is stored in detector-name order.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformConfig
    network: Network
    poisson_rate: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    num_segments: Annotated[int, Field(ge=1)]
    segment_duration: Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
    simulate_detector_noise: bool = False
    population: str
    population_root: str = "source_parameters"
    seed: int

    @field_validator("network")
    @classmethod
    def _validate_network(cls, network: Network) -> Network:
        network_pairs(network)
        return dict(sorted(network.items()))

    def to_metadata(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible document.

        The network includes the full detector definitions.
        :meth:`from_metadata` inverts it.

        Returns
        -------
        dict
            The JSON-compatible document.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> SimulationConfig:
        """Rebuild a config from the document written by :meth:`to_metadata`.

        Parameters
        ----------
        metadata
            The JSON-compatible document.

        Returns
        -------
        SimulationConfig
            The resolved config, independent of any detector registration.
        """
        return cls.model_validate(metadata)

    @property
    def observation_seconds(self) -> float:
        """The observed time in seconds: ``num_segments * segment_duration``."""
        return self.num_segments * self.segment_duration
