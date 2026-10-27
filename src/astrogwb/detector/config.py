"""Serializable detector settings, resolved before runtime objects are built.

Angles use gwmock's radians (azimuths clockwise from North); elevations use
metres. Importing this module loads the detector package, including JAX.
Validating settings leaves the XLA backend uninitialized.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import knf
from gwmock_signal.detector import CustomDetector
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .psd import resolve_psd_path
from .sensitivity import Sensitivity

_MERGE_KEY = "extends"
#: The packaged detector geometry and sensitivity tables, the first layers of
#: every merge; they use the same tables as the layers that override them.
DETECTOR_DEFAULT_PATHS = tuple(
    Path(__file__).parent / filename
    for filename in ("geometry.toml", "sensitivity.toml")
)

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
logger = logging.getLogger(__name__)


class DetectorGeometry(BaseModel):
    """Complete geometry in gwmock radians and metres, matching ``geometry.toml``."""

    model_config = _STRICT

    latitude_rad: Annotated[float, Field(ge=-math.pi / 2.0, le=math.pi / 2.0)]
    longitude_rad: Annotated[float, Field(ge=-math.pi, le=math.pi)]
    elevation_m: Annotated[float, Field(ge=-1e4, le=1e5)]
    xarm_azimuth_rad: float
    yarm_azimuth_rad: float
    xarm_tilt_rad: float = 0.0
    yarm_tilt_rad: float = 0.0


class DetectorConfig(BaseModel):
    """Geometry, PSD reference, and an optional presentation label.

    A :class:`DetectorRegistry` labels an unlabelled detector with its name.
    """

    model_config = _STRICT

    geometry: DetectorGeometry
    psd_reference: Annotated[str, Field(min_length=1, pattern=r"\S")]
    label: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _validate_reference(self) -> DetectorConfig:

        try:
            resolved = resolve_psd_path(self.psd_reference)
            if isinstance(resolved, Path) and not resolved.is_file():
                raise ValueError(
                    f"PSD reference must be a file: {self.psd_reference!r}"
                )
        except FileNotFoundError as error:
            raise ValueError(str(error)) from error
        return self

    @property
    def sensitivity(self) -> Sensitivity:
        """Sensitivity built from this detector's PSD reference."""
        return Sensitivity(self.psd_reference)

    def to_detector(self, name: str) -> CustomDetector:
        """gwmock geometry object addressed by ``name``."""
        return CustomDetector(name=name, **self.geometry.model_dump())


class DetectorRegistry(BaseModel):
    """A complete resolved registry, trusted and fixed for a dataset's lifetime.

    Each network's members are stored sorted, the canonical order of
    :meth:`pairs`, so pair axes built from the registry line up with a
    simulated background's. The order they were declared in is not kept, and
    the ``detectors`` and ``networks`` mappings are not to be edited after
    construction.
    """

    model_config = _STRICT

    detectors: dict[str, DetectorConfig]
    networks: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @field_validator("networks")
    @classmethod
    def _canonical_order(
        cls, networks: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        return {name: tuple(sorted(members)) for name, members in networks.items()}

    @field_validator("detectors")
    @classmethod
    def _label_detectors(
        cls, detectors: dict[str, DetectorConfig]
    ) -> dict[str, DetectorConfig]:
        """Label each unlabelled detector with its name, on a copy."""
        for name in detectors:
            if not name.strip():
                raise ValueError("detector names must be non-empty")
        return {
            name: detector
            if detector.label is not None
            else detector.model_copy(update={"label": name})
            for name, detector in detectors.items()
        }

    @model_validator(mode="after")
    def _validate_networks(self) -> DetectorRegistry:
        for name, members in self.networks.items():
            self.validate_members(members, label=f"network {name!r}")
        return self

    def validate_members(
        self, names: Sequence[str], *, label: str = "detectors"
    ) -> None:
        """Require a non-empty, distinct list of complete detector definitions.

        Parameters
        ----------
        names
            The detector names to check.
        label
            What ``names`` is, for the error message.

        Raises
        ------
        ValueError
            If ``names`` is empty, repeats a detector, or names one the
            registry does not define.
        """
        if not names or len(set(names)) != len(names):
            raise ValueError(
                f"{label} must contain distinct detectors and be non-empty"
            )
        missing = [name for name in names if name not in self.detectors]
        if missing:
            raise ValueError(f"{label} contains undefined detectors: {missing}")

    def build_detectors(
        self, names: Sequence[str]
    ) -> tuple[tuple[CustomDetector, ...], dict[str, Sensitivity]]:
        """Build selected geometry and sensitivities in the supplied order.

        Parameters
        ----------
        names
            The detectors to build; see :meth:`validate_members`.

        Returns
        -------
        tuple
            The detectors, in the order of ``names``, and their sensitivities
            keyed by name.
        """
        self.validate_members(names)
        return (
            tuple(self.detectors[name].to_detector(name) for name in names),
            {name: self.detectors[name].sensitivity for name in names},
        )

    def build_network(
        self, name: str
    ) -> tuple[tuple[CustomDetector, ...], dict[str, Sensitivity]]:
        """Build a named network, its detectors sorted.

        Unlike :meth:`build_detectors`, the order is the canonical one of the
        network, not the order it was declared in.

        Parameters
        ----------
        name
            A network of the registry.

        Returns
        -------
        tuple
            The detectors and their sensitivities keyed by name.
        """
        return self.build_detectors(self.networks[name])

    def pairs(self, network_name: str) -> list[tuple[str, str]]:
        """Return the canonical sorted detector pairs of a named network.

        Parameters
        ----------
        network_name
            A network of the registry.

        Returns
        -------
        list of tuple of str
            Each pair of members, ``(a, b)`` with ``a`` before ``b``.
        """
        detectors = self.networks[network_name]
        return [(a, b) for i, a in enumerate(detectors) for b in detectors[i + 1 :]]


def load_detector_config(paths: Sequence[Path]) -> DetectorRegistry:
    """Merge packaged detector definitions and ordered TOML registry layers.

    Later layers override earlier geometry, sensitivities and network
    membership. References and ``extends`` resolve within this merge alone.
    An empty layer list loads packaged detectors without named networks.
    Only ``[detectors]`` and ``[networks]`` are accepted in the resolved config;
    no runtime detector objects are built during loading or validation.

    Parameters
    ----------
    paths
        Existing TOML files, merged in order after the packaged defaults.

    Returns
    -------
    DetectorRegistry
        The validated registry.

    Raises
    ------
    ValueError
        If a path is not an existing TOML file, or the merged config is not a
        valid registry.
    """
    for path in paths:
        if path.suffix.lower() != ".toml" or not path.is_file():
            raise ValueError(f"detector config must be an existing TOML file: {path}")
    logger.info(
        "Detector config layers (merge order): %s",
        " -> ".join(str(path) for path in (*DETECTOR_DEFAULT_PATHS, *paths)),
    )
    merged = knf.load(
        [*DETECTOR_DEFAULT_PATHS, *paths],
        interpolate=True,
        merge_key=_MERGE_KEY,
    )
    return DetectorRegistry.model_validate(merged)
