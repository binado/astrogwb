"""Detector geometry sourced from astrogwb's ``geometry.toml`` table.

gwmock ``CustomDetector`` presets carry their own geometry; astrogwb adds a
local table for LAL/CE site codes (plain strings) and other detectors that
do not ship as gwmock presets.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from functools import cache
from pathlib import Path

from gwmock_signal.detector import CustomDetector

from ._types import DetectorSpec

GEOMETRY_FILE = Path(__file__).parent / "geometry.toml"


@cache
def _geometry_table() -> Mapping[str, CustomDetector]:
    """Load ``geometry.toml`` into name -> ``CustomDetector`` (radians)."""
    with open(GEOMETRY_FILE, "rb") as f:
        data = tomllib.load(f)
    return {name: _custom_detector(name, row) for name, row in data.items()}


def _custom_detector(name: str, row: Mapping) -> CustomDetector:
    return CustomDetector(
        name=name,
        latitude_rad=math.radians(float(row["latitude"])),
        longitude_rad=math.radians(float(row["longitude"])),
        elevation_m=float(row["elevation"]),
        xarm_azimuth_rad=math.radians(float(row["xarm_azimuth"])),
        yarm_azimuth_rad=math.radians(float(row["yarm_azimuth"])),
        xarm_tilt_rad=math.radians(float(row.get("xarm_tilt", 0.0))),
        yarm_tilt_rad=math.radians(float(row.get("yarm_tilt", 0.0))),
    )


def load_geometry(name: str) -> CustomDetector:
    """Look up a detector's geometry from the ``geometry.toml`` table.

    Returns a gwmock ``CustomDetector``, which can be fed to
    :meth:`Network.from_detectors` to compose custom networks (e.g. mixing
    LAL site codes with non-LAL detectors such as Cosmic Explorer at
    Livingston, which has no LAL code).
    """
    try:
        return _geometry_table()[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown detector {name!r}; not in geometry.toml. "
            "Pass a gwmock CustomDetector for ad-hoc geometry."
        ) from exc


def resolve_geometry(spec: DetectorSpec) -> CustomDetector:
    """Resolve a detector spec to a ``CustomDetector`` carrying geometry.

    ``CustomDetector`` instances (ET presets, ad-hoc configs) already carry
    angles and pass through; ``str`` LAL/CE/ET site codes are looked up in
    the :mod:`geometry.toml` table.
    """
    if isinstance(spec, CustomDetector):
        return spec
    return load_geometry(spec)
