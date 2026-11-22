"""The inputs that determine one simulated background, recorded with its file."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrogwb.detector import Detector, Networks
from astrogwb.detector.config import _DetectorConfigDocument
from astrogwb.waveform.metadata import WaveformConfig

__all__ = ["SimulationConfig"]


def _network_detectors(
    networks: Networks,
) -> dict[str, Detector]:
    """Resolve the shared definitions and validate that the networks can run together."""
    if not networks:
        raise ValueError("a background needs at least one network")
    detectors: dict[str, Detector] = {}
    for network_name, network in networks.items():
        if not network_name.strip():
            raise ValueError("network names must be non-blank")
        if len(network) < 2:
            raise ValueError(
                f"network {network_name!r} must contain at least two detectors"
            )
        for name, detector in network.items():
            if name != detector.name:
                raise ValueError(
                    f"detector name {detector.name!r} does not match mapping key {name!r}"
                )
            previous = detectors.setdefault(name, detector)
            if previous != detector:
                raise ValueError(f"conflicting definitions for detector {name!r}")
    return detectors


class SimulationConfig(BaseModel):
    """Everything that determines a simulated background, but its draw.

    ``waveform`` and ``networks`` set how events are projected onto the
    networks. ``poisson_rate`` is the rate, in events per second, of the
    population the event pool represents. The observation is ``num_segments``
    segments of ``segment_seconds`` each. With ``incoherent`` the spectra are
    the noiseless sums of event powers; otherwise each segment adds coherent
    strain and detector noise (see :mod:`astrogwb.background.simulation.draw`).

    ``population``, ``population_root`` and ``seed`` record where the pool came
    from and the key the draw was made at. The root is ignored for CSV files.
    They are provenance: they do not change how
    the spectra are computed. Execution details such as chunk sizes change cost,
    never the result, and are not recorded.

    ``networks`` are held sorted by name, so configs that differ only in the
    order the networks were given compare equal.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    waveform: WaveformConfig
    networks: Networks
    poisson_rate: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    num_segments: Annotated[int, Field(ge=1)]
    segment_seconds: Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
    incoherent: bool = False
    population: str
    population_root: str = "source_parameters"
    seed: int

    @field_validator("networks")
    @classmethod
    def _validate_networks(cls, networks: Networks) -> Networks:
        _network_detectors(networks)
        return {
            name: dict(sorted(network.items()))
            for name, network in sorted(networks.items())
        }

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
                name: tuple(sorted(network)) for name, network in self.networks.items()
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
        if "observer_rate_hz" in settings:
            settings["poisson_rate"] = settings.pop("observer_rate_hz")
        document = _DetectorConfigDocument.model_validate(settings.pop("detectors"))
        return cls.model_validate({**settings, "networks": document.resolve_networks()})

    @property
    def observation_seconds(self) -> float:
        """The observed time in seconds: ``num_segments * segment_seconds``."""
        return self.num_segments * self.segment_seconds
