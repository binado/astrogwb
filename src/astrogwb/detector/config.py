"""Detector definitions and network selection.

Angles use radians (azimuths clockwise from North); elevations use
metres. Importing this module loads the detector package, including JAX.
Validating settings leaves the XLA backend uninitialized.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping, Sequence
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
    """Shared detector definitions in TOML and metadata."""

    model_config = _STRICT

    detectors: _Detectors


def load_detectors_from_file(path: Path) -> dict[str, Detector]:
    """Load every detector definition from a complete TOML document.

    Parameters
    ----------
    path
        Path to a TOML file with one ``[detectors.<name>]`` table per detector.

    Returns
    -------
    dict
        Detectors keyed by name.
    """
    with path.open("rb") as file:
        document = _DetectorConfigDocument.model_validate(tomllib.load(file))
    return document.detectors


def select_network(detectors: Mapping[str, Detector], names: Sequence[str]) -> Network:
    """Select the detectors that form a network.

    Parameters
    ----------
    detectors
        Available detectors keyed by name.
    names
        Names of the detectors in the network, in any order.

    Returns
    -------
    Network
        The selected detectors keyed by name, in sorted name order.

    Raises
    ------
    ValueError
        If a name is repeated or is not among the available detectors.
    """
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate detector names in network: {list(names)}")
    missing = sorted(set(names) - detectors.keys())
    if missing:
        available = ", ".join(sorted(detectors)) or "(none)"
        raise ValueError(
            f"unknown detectors {missing}. Available detectors: {available}"
        )
    return {name: detectors[name] for name in sorted(names)}


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
