"""Detector definitions and named network settings.

Angles use radians (azimuths clockwise from North); elevations use
metres. Importing this module loads the detector package, including JAX.
Validating settings leaves the XLA backend uninitialized.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    WrapSerializer,
    model_validator,
)

from .geometry import Detector

#: The packaged detector geometry and sensitivity tables, the first layers of
#: every merge; they use the same tables as the layers that override them.
_DETECTOR_DEFAULT_PATHS = tuple(
    Path(__file__).parent / filename
    for filename in ("geometry.toml", "sensitivity.toml")
)

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_logger = logging.getLogger(__name__)
_Name = Annotated[str, Field(min_length=1, pattern=r"\S")]


def _name_detectors(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value

    detectors = {}
    for name, definition in value.items():
        if isinstance(definition, Detector):
            explicit_name = definition.name
        elif isinstance(definition, Mapping):
            explicit_name = definition.get("name", name)
            definition = {**definition, "name": explicit_name}
        else:
            explicit_name = name
        if explicit_name != name:
            raise ValueError(
                f"detector name {explicit_name!r} does not match mapping key {name!r}"
            )
        detectors[name] = definition
    return detectors


def _dump_detectors(
    detectors: dict[str, Detector], handler: SerializerFunctionWrapHandler
) -> dict[str, Any]:
    definitions = handler(detectors)
    for definition in definitions.values():
        definition.pop("name", None)
    return definitions


_Detectors = Annotated[
    dict[_Name, Detector],
    BeforeValidator(_name_detectors),
    WrapSerializer(_dump_detectors),
]


Network = Mapping[str, Detector]
Networks = Mapping[str, Network]


def network_pairs(network: Network) -> tuple[tuple[str, str], ...]:
    """Return detector key pairs in the order of spectra and noise rows."""
    return tuple(combinations(sorted(network), 2))


class _DetectorConfigDocument(BaseModel):
    """Shared detector definitions and named memberships in TOML and metadata."""

    model_config = _STRICT

    detectors: _Detectors
    networks: dict[_Name, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_networks(self) -> _DetectorConfigDocument:
        for name, members in self.networks.items():
            if not members or len(set(members)) != len(members):
                raise ValueError(
                    f"network {name!r} must contain distinct detectors and be non-empty"
                )
            missing = sorted(set(members) - self.detectors.keys())
            if missing:
                raise ValueError(
                    f"network {name!r} contains undefined detectors: {missing}"
                )
        return self

    @classmethod
    def from_files(cls, *paths: Path) -> _DetectorConfigDocument:
        from astrogwb.config import merge_layers

        layers = (*_DETECTOR_DEFAULT_PATHS, *paths)
        _logger.info(
            "Detector config layers (merge order): %s", " -> ".join(map(str, layers))
        )
        return cls.model_validate(merge_layers(layers))

    def resolve_networks(self) -> Networks:
        return {
            name: {member: self.detectors[member] for member in sorted(members)}
            for name, members in sorted(self.networks.items())
        }


def load_networks_from_files(*paths: Path) -> Networks:
    """Resolve all named networks from ordered TOML layers.

    Parameters
    ----------
    paths
        Layers applied after the packaged detector definitions. The process
        registry is independent of these layers.
    """
    return _DetectorConfigDocument.from_files(*paths).resolve_networks()


_NETWORKS: dict[str, frozenset[str]] = {}


@cache
def _detector_catalog() -> dict[str, Detector]:
    return _DetectorConfigDocument.from_files().detectors


def register_detector(detector: Detector) -> None:
    """Add a detector to the process catalog; existing names cannot be replaced.

    Parameters
    ----------
    detector
        The detector to register.
    """
    catalog = _detector_catalog()
    name = detector.name
    if name in catalog:
        raise ValueError(f"detector {name!r} is already registered")
    catalog[name] = detector


def register_network(name: str, members: AbstractSet[str]) -> None:
    """Register a non-empty set of detector names without replacing an existing network.

    Parameters
    ----------
    name
        Non-blank network name.
    members
        Names of already registered detectors.
    """
    if not name.strip():
        raise ValueError("network names must be non-empty")
    if name in _NETWORKS:
        raise ValueError(f"network {name!r} is already registered")
    if not members:
        raise ValueError(f"network {name!r} must be non-empty")
    missing = sorted(members - _detector_catalog().keys())
    if missing:
        raise ValueError(f"network {name!r} contains undefined detectors: {missing}")
    _NETWORKS[name] = frozenset(members)


def load_detector(name: str) -> Detector:
    """Load a registered detector without reading its PSD curve.

    Parameters
    ----------
    name
        Name to look up.
    """
    return _detector_catalog()[name]


def load_network(name: str) -> Network:
    """Return a fresh mapping of registered detectors keyed by sorted name.

    Parameters
    ----------
    name
        Name of a registered network.
    """
    return {member: load_detector(member) for member in sorted(_NETWORKS[name])}


def resolve_network(network: str | Sequence[Detector]) -> tuple[Detector, ...]:
    """Resolve a network name or detector sequence to detectors.

    Parameters
    ----------
    network
        A registered network name, loaded in canonical sorted order, or
        detectors, kept in the given order.
    """
    if isinstance(network, str):
        return tuple(load_network(network).values())
    return tuple(network)
