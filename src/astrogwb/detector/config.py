"""Detector definitions and named network settings.

Angles use radians (azimuths clockwise from North); elevations use
metres. Importing this module loads the detector package, including JAX.
Validating settings leaves the XLA backend uninitialized.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
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

_DETECTOR_DATA = Path(__file__).parent

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
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


#: Detectors keyed by their own name.
Network = Mapping[str, Detector]
Networks = Mapping[str, Network]


def network_pairs(network: Network) -> tuple[tuple[str, str], ...]:
    """Return detector key pairs in the order of spectra and noise rows.

    Parameters
    ----------
    network
        Detectors keyed by name.

    Returns
    -------
    tuple
        Every pair of keys, each sorted, in lexicographic order.

    Raises
    ------
    ValueError
        If the network has fewer than two detectors, or a key differs from the
        name of its detector.
    """
    if len(network) < 2:
        raise ValueError("a network needs at least two detectors to form a pair")
    for key, detector in network.items():
        if detector.name != key:
            raise ValueError(
                f"detector {detector.name!r} is keyed by {key!r} in the network"
            )
    return tuple(combinations(sorted(network), 2))


def map_over_pairs[T](
    function: Callable[[Detector, Detector], T], network: Network
) -> dict[tuple[str, str], T]:
    """Evaluate a function on each pair in canonical detector-name order.

    Parameters
    ----------
    function
        Function receiving the two detectors in name order.
    network
        Detectors keyed by name.

    Returns
    -------
    dict
        Function results keyed by detector-name pair, in network pair order.
    """
    return {
        (first, second): function(network[first], network[second])
        for first, second in network_pairs(network)
    }


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

    def resolve_networks(self) -> Networks:
        return {
            name: {member: self.detectors[member] for member in sorted(members)}
            for name, members in sorted(self.networks.items())
        }


def load_networks_from_file(path: Path) -> Networks:
    """Resolve all named networks from a complete TOML document.

    Parameters
    ----------
    path
        Path to a TOML file containing the network memberships and all detector
        definitions they reference.
    """
    with path.open("rb") as file:
        document = _DetectorConfigDocument.model_validate(tomllib.load(file))
    return document.resolve_networks()


_NETWORKS: dict[str, frozenset[str]] = {}


@cache
def _detector_catalog() -> dict[str, Detector]:
    with (_DETECTOR_DATA / "geometry.toml").open("rb") as file:
        geometry = tomllib.load(file)["detectors"]
    with (_DETECTOR_DATA / "sensitivity.toml").open("rb") as file:
        sensitivity = tomllib.load(file)["detectors"]
    detectors = {name: fields | sensitivity[name] for name, fields in geometry.items()}
    return _DetectorConfigDocument(detectors=detectors).detectors


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
