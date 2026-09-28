"""Serializable detector settings, resolved before runtime objects are built.

Angles use the packaged table's degrees and elevations use metres. Only the
build methods import detector libraries; reading and validating settings does
not import JAX or initialize its backend.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb.paper.utils import deep_merge

if TYPE_CHECKING:
    from gwmock_signal.detector import CustomDetector

    from astrogwb.detector import Sensitivity

_STRICT = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
_PACKAGED = Path(__file__).parents[2] / "detector"


class DetectorGeometry(BaseModel):
    """Complete geometry in degrees and metres, matching ``geometry.toml``."""

    model_config = _STRICT

    latitude: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]
    elevation: float
    xarm_azimuth: float
    yarm_azimuth: float
    xarm_tilt: float = 0.0
    yarm_tilt: float = 0.0

    def build(self, name: str) -> CustomDetector:
        """Construct geometry lazily, preserving the public detector name."""
        from gwmock_signal.detector import CustomDetector

        return CustomDetector(
            name=name,
            latitude_rad=math.radians(self.latitude),
            longitude_rad=math.radians(self.longitude),
            elevation_m=self.elevation,
            xarm_azimuth_rad=math.radians(self.xarm_azimuth),
            yarm_azimuth_rad=math.radians(self.yarm_azimuth),
            xarm_tilt_rad=math.radians(self.xarm_tilt),
            yarm_tilt_rad=math.radians(self.yarm_tilt),
        )


class DetectorConfig(BaseModel):
    """Geometry, PSD reference, and an optional presentation label."""

    model_config = _STRICT

    geometry: DetectorGeometry
    psd_reference: Annotated[str, Field(min_length=1, pattern=r"\S")]
    label: Annotated[str, Field(min_length=1)] | None = None

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

    def build_sensitivity(self) -> Sensitivity:
        """Build a sensitivity using the existing preset/file/URL resolver."""
        from astrogwb.detector import Sensitivity
        from astrogwb.psd import resolve_psd_path

        resolve_psd_path(self.psd_reference)
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

    @classmethod
    def from_overrides(
        cls,
        detectors: Mapping[str, Any] | None = None,
        networks: Mapping[str, Sequence[str]] | None = None,
    ) -> DetectorRegistry:
        """Merge packaged defaults field by field, then validate complete settings.

        Only names with both packaged geometry and sensitivity are defaults.
        New names must supply both; arm tilts default to zero.
        """
        geometry = tomllib.loads((_PACKAGED / "geometry.toml").read_text())
        sensitivity = tomllib.loads((_PACKAGED / "sensitivity.toml").read_text())
        defaults = {
            name: {
                "geometry": row,
                "psd_reference": sensitivity[name]["psd_reference"],
            }
            for name, row in geometry.items()
            if name in sensitivity
        }
        return cls.model_validate(
            {
                "detectors": deep_merge(defaults, dict(detectors or {})),
                "networks": dict(networks or {}),
            }
        )

    def build_detectors(
        self, names: Sequence[str]
    ) -> tuple[tuple[CustomDetector, ...], dict[str, Sensitivity]]:
        """Build selected geometry and sensitivities in the supplied order."""
        self.validate_members(names)
        return (
            tuple(self.detectors[name].geometry.build(name) for name in names),
            {name: self.detectors[name].build_sensitivity() for name in names},
        )

    def build_network(
        self, name: str
    ) -> tuple[tuple[CustomDetector, ...], dict[str, Sensitivity]]:
        """Build a named network in its declared detector order."""
        return self.build_detectors(self.networks[name])

    def local_psd_inputs(self, names: Sequence[str]) -> tuple[Path, ...]:
        """Selected external local PSD files, using the runtime resolution order.

        Presets and packaged files travel with the installed library, while
        external files need explicit workflow edges. URLs are loaded at runtime.
        """
        from astrogwb.psd import NOISE_CURVES_BASE_DIR, resolve_psd_path

        self.validate_members(names)
        paths: dict[Path, None] = {}
        for name in names:
            resolved = resolve_psd_path(self.detectors[name].psd_reference)
            if isinstance(resolved, Path):
                absolute = resolved.resolve()
                # A preset wins even if a same-named local file exists.
                if absolute == Path(
                    self.detectors[name].psd_reference
                ).resolve() and not absolute.is_relative_to(
                    NOISE_CURVES_BASE_DIR.resolve()
                ):
                    paths[resolved] = None
        return tuple(paths)
