"""The inputs that determine one simulated background, recorded with its file."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrogwb.detector import DetectorConfig, DetectorNetworkConfig
from astrogwb.detector.config import _DetectorConfigDocument
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

    ``networks`` are held sorted by name, so configs that differ only in the
    order the networks were given compare equal.
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
        return tuple(sorted(networks, key=lambda network: network.name))

    def to_metadata(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible document.

        The settings are stored flat. The networks are stored as one ``detectors``
        document: each detector definition once, plus the member names of every
        network. :meth:`from_metadata` inverts it.

        Returns
        -------
        dict
            The JSON-compatible document.
        """
        document = _DetectorConfigDocument(
            detectors=_network_detectors(self.networks),
            networks={
                network.name: tuple(sorted(network.detectors))
                for network in self.networks
            },
        )
        metadata = self.model_dump(mode="json", exclude={"networks"})
        metadata["detectors"] = document.model_dump(mode="json")
        return metadata

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
        settings = dict(metadata)
        document = _DetectorConfigDocument.model_validate(settings.pop("detectors"))
        return cls.model_validate(
            {
                **settings,
                "networks": tuple(
                    document.select_network(name) for name in document.networks
                ),
            }
        )

    @property
    def observation_seconds(self) -> float:
        """The observed time in seconds: ``num_segments * segment_seconds``."""
        return self.num_segments * self.segment_seconds
