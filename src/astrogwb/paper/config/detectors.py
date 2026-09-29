"""Serializable detector settings, resolved before runtime objects are built.

Angles use gwmock's radians (azimuths clockwise from North); elevations use
metres. Importing this module loads the detector package, including JAX.
Validating settings leaves the XLA backend uninitialized.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from gwmock_signal.detector import CustomDetector
from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb.detector import Sensitivity

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


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
    """Geometry, PSD reference, and an optional presentation label."""

    model_config = _STRICT

    geometry: DetectorGeometry
    psd_reference: Annotated[str, Field(min_length=1, pattern=r"\S")]
    label: Annotated[str, Field(min_length=1)] | None = None
    #: Reference metadata; does not rescale the PSD or observation time.
    duty_factor: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def _validate_reference(self) -> DetectorConfig:
        from astrogwb.psd import resolve_psd_path

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


class DetectorRegistry(BaseModel):
    """A complete resolved registry; serialized records need no default merge."""

    model_config = _STRICT

    detectors: dict[str, DetectorConfig]
    networks: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_networks(self) -> DetectorRegistry:
        for name, detector in self.detectors.items():
            if not name.strip():
                raise ValueError("detector names must be non-empty")
            if detector.label is None:
                object.__setattr__(detector, "label", name)
        for name, members in self.networks.items():
            self.validate_members(members, label=f"network {name!r}")
        return self

    def validate_members(
        self, names: Sequence[str], *, label: str = "detectors"
    ) -> None:
        """Require a non-empty, distinct list of complete detector definitions."""
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
        """Build selected geometry and sensitivities in the supplied order."""
        self.validate_members(names)
        return (
            tuple(
                CustomDetector(name=name, **self.detectors[name].geometry.model_dump())
                for name in names
            ),
            {name: self.detectors[name].sensitivity for name in names},
        )

    def build_network(
        self, name: str
    ) -> tuple[tuple[CustomDetector, ...], dict[str, Sensitivity]]:
        """Build a named network in its declared detector order."""
        return self.build_detectors(self.networks[name])
