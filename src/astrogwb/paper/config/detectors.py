"""Serializable detector settings, resolved before runtime objects are built.

Angles use gwmock's radians (azimuths clockwise from North); elevations use
metres. Only the build methods import detector libraries; validating settings does
not import JAX or initialize its backend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from gwmock_signal.detector import CustomDetector

    from astrogwb.detector import Sensitivity

_LEGACY_GEOMETRY = frozenset(
    {
        "latitude",
        "longitude",
        "elevation",
        "xarm_azimuth",
        "yarm_azimuth",
        "xarm_tilt",
        "yarm_tilt",
    }
)

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

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and (legacy := data.keys() & _LEGACY_GEOMETRY):
            raise ValueError(
                f"legacy geometry fields {sorted(legacy)} are only supported in "
                "saved JSON registries; use gwmock _rad/_m fields for overrides"
            )
        return data


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

    @classmethod
    def from_saved(cls, data: Any) -> DetectorRegistry:
        """Restore saved settings, translating pre-gwmock degree geometry.

        This migration is exclusive to saved registries, never TOML overrides.
        Copy each translated record so the caller's saved mapping stays intact.
        """
        if not isinstance(data, Mapping):
            return cls.model_validate(data)
        restored = dict(data)
        definitions = restored.get("detectors")
        if isinstance(definitions, Mapping):
            definitions = dict(definitions)
            for name, record in definitions.items():
                if not isinstance(record, Mapping):
                    continue
                geometry = record.get("geometry")
                if not isinstance(geometry, Mapping):
                    continue
                if not geometry.keys() & _LEGACY_GEOMETRY:
                    continue
                if any(key.endswith(("_rad", "_m")) for key in geometry):
                    raise ValueError(
                        f"mixed legacy and gwmock geometry fields for {name!r}"
                    )
                translated = {}
                for key, value in geometry.items():
                    if key == "elevation":
                        translated["elevation_m"] = value
                    elif key in _LEGACY_GEOMETRY:
                        angle = math.radians(value)
                        if "azimuth" in key:
                            angle = (math.pi / 2.0 - angle) % (2.0 * math.pi)
                        translated[f"{key}_rad"] = angle
                    else:
                        translated[key] = value
                definitions[name] = dict(record, geometry=translated)
            restored["detectors"] = definitions
        return cls.model_validate(restored)

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
        from gwmock_signal.detector import CustomDetector

        self.validate_members(names)
        return (
            tuple(
                CustomDetector(name=name, **self.detectors[name].geometry.model_dump())
                for name in names
            ),
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
