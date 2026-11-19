"""Serializable detector settings, resolved before runtime objects are built.

Angles use radians (azimuths clockwise from North); elevations use
metres. Importing this module loads the detector package, including JAX.
Validating settings leaves the XLA backend uninitialized.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .geometry import Detector, DetectorGeometry
from .psd import resolve_psd_path
from .sensitivity import Sensitivity

#: The packaged detector geometry and sensitivity tables, the first layers of
#: every merge; they use the same tables as the layers that override them.
DETECTOR_DEFAULT_PATHS = tuple(
    Path(__file__).parent / filename
    for filename in ("geometry.toml", "sensitivity.toml")
)

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
logger = logging.getLogger(__name__)
_Name = Annotated[str, Field(min_length=1, pattern=r"\S")]


@cache
def _sensitivity(psd_reference: str) -> Sensitivity:
    """The shared :class:`Sensitivity` of a PSD reference."""
    return Sensitivity(psd_reference)


class DetectorConfig(BaseModel):
    """Geometry, PSD reference, and an optional presentation label."""

    model_config = _STRICT

    geometry: DetectorGeometry
    psd_reference: Annotated[str, Field(min_length=1, pattern=r"\S")]
    label: Annotated[str, Field(min_length=1)] | None = None

    @field_validator("psd_reference")
    @classmethod
    def _validate_reference(cls, reference: str) -> str:
        try:
            resolved = resolve_psd_path(reference)
            if isinstance(resolved, Path) and not resolved.is_file():
                raise ValueError(f"PSD reference must be a file: {reference!r}")
        except FileNotFoundError as error:
            raise ValueError(str(error)) from error
        return reference

    @property
    def sensitivity(self) -> Sensitivity:
        """Sensitivity of this detector's PSD reference, cached per reference.

        Caching per reference, not per instance, keeps ``model_copy`` from
        carrying a stale curve.
        """
        return _sensitivity(self.psd_reference)

    def to_detector(self, name: str) -> Detector:
        """Build an owned detector addressed by ``name``.

        Parameters
        ----------
        name
            Name given to the built detector.
        """
        return Detector(name, self.geometry, self.sensitivity)


class DetectorNetworkConfig(BaseModel):
    """One named network with its resolved detector definitions.

    The keys of ``detectors`` are the members. Runtime detectors and pairs
    follow sorted detector names, independent of declaration order.
    """

    model_config = _STRICT

    name: _Name
    detectors: Annotated[dict[_Name, DetectorConfig], Field(min_length=1)]

    @classmethod
    def from_network(cls, name: str) -> DetectorNetworkConfig:
        """Snapshot a registered network.

        The result is unaffected by later registrations, because the detector
        definitions are immutable.

        Parameters
        ----------
        name
            Name of a registered network.
        """
        members = _NETWORKS[name]
        catalog = _detector_catalog()
        return cls(
            name=name,
            detectors={member: catalog[member] for member in sorted(members)},
        )

    @classmethod
    def from_files(cls, *paths: Path, network: str) -> DetectorNetworkConfig:
        """Select one network from ordered TOML layers, independent of the catalog.

        Parameters
        ----------
        paths
            Layers applied after the packaged detector definitions.
        network
            Name of a network in the merged document.
        """
        return _DetectorConfigDocument.from_files(*paths).select_network(network, cls)

    def build_detectors(self) -> tuple[Detector, ...]:
        """Build runtime detectors in canonical sorted order."""
        return tuple(
            self.detectors[name].to_detector(name) for name in sorted(self.detectors)
        )

    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Return the canonical pair axis, with the earlier name first in each pair."""
        return tuple(combinations(sorted(self.detectors), 2))


class _DetectorConfigDocument(BaseModel):
    """Shared detector definitions and named memberships in TOML and metadata."""

    model_config = _STRICT

    detectors: dict[_Name, DetectorConfig]
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

        layers = (*DETECTOR_DEFAULT_PATHS, *paths)
        logger.info(
            "Detector config layers (merge order): %s", " -> ".join(map(str, layers))
        )
        return cls.model_validate(merge_layers(layers))

    def select_network(
        self, name: str, model: type[DetectorNetworkConfig] = DetectorNetworkConfig
    ) -> DetectorNetworkConfig:
        return model(
            name=name,
            detectors={
                member: self.detectors[member] for member in sorted(self.networks[name])
            },
        )


_NETWORKS: dict[str, frozenset[str]] = {}


@cache
def _detector_catalog() -> dict[str, DetectorConfig]:
    return _DetectorConfigDocument.from_files().detectors


def register_detector(name: str, config: DetectorConfig) -> None:
    """Add a detector to the process catalog; existing names cannot be replaced.

    Parameters
    ----------
    name
        Non-blank detector name.
    config
        The resolved detector definition.
    """
    catalog = _detector_catalog()
    if not name.strip():
        raise ValueError("detector names must be non-empty")
    if name in catalog:
        raise ValueError(f"detector {name!r} is already registered")
    catalog[name] = config


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


@cache
def load_detector(name: str) -> Detector:
    """Load a registered detector without reading its PSD curve.

    Parameters
    ----------
    name
        Name to look up.
    """
    return _detector_catalog()[name].to_detector(name)


def load_network(name: str) -> tuple[Detector, ...]:
    """Load the detectors of a registered network in canonical sorted order.

    Parameters
    ----------
    name
        Name of a registered network.
    """
    return tuple(load_detector(member) for member in sorted(_NETWORKS[name]))


def resolve_network(network: str | Sequence[Detector]) -> tuple[Detector, ...]:
    """Resolve a network name or detector sequence to detectors.

    Parameters
    ----------
    network
        A registered network name, loaded in canonical sorted order, or
        detectors, kept in the given order.
    """
    return load_network(network) if isinstance(network, str) else tuple(network)
