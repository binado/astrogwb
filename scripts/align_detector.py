"""Rotate an L-shaped detector to a target alignment with another detector.

Prints a ``geometry.toml`` block for the rotated detector, keeping its site,
elevation and opening angle, so that its arm bisector sits at ``--angle``
degrees from the reference detector's. The angle is measured either from the
great circle joining the sites (the convention of the overlap reduction
function) or from local North at each site (the angle alpha of Branchesi et
al. 2023, arXiv:2303.15923). For example, the misaligned ET 2L pair::

    uv run python scripts/align_detector.py S2 R2 --angle 45 --reference north

A summary of the resulting angles and the long-wavelength overlap reduction
function goes to stderr, so the block can be redirected on its own.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import get_args

from gwmock_signal.detector import CustomDetector

from astrogwb.detector import (
    OrientationReference,
    align_detector,
    overlap_reduction_function,
    relative_orientation,
)

#: Frequency (Hz) at which the summary reports the overlap reduction function;
#: well inside the long-wavelength limit for ET-scale baselines.
SUMMARY_FREQUENCY_HZ = 1.0


def _from_east_degrees(azimuth_rad: float) -> float:
    """Convert a clockwise-from-North azimuth to degrees counter-clockwise from East."""
    return math.degrees(math.pi / 2.0 - azimuth_rad) % 360.0


def _arm_line(key: str, azimuth_rad: float) -> str:
    """One ``geometry.toml`` arm azimuth with its counter-clockwise-from-East comment."""
    degrees = _from_east_degrees(azimuth_rad)
    return (
        f"{key} = {azimuth_rad!r}"
        f"  # Original: {degrees:.10g} deg, counter-clockwise from East"
    )


def geometry_block(detector: CustomDetector, name: str) -> str:
    """Format a detector as a ``geometry.toml`` table."""
    return "\n".join(
        [
            f"[detectors.{name}.geometry]",
            f"latitude_rad = {detector.latitude_rad!r}",
            f"longitude_rad = {detector.longitude_rad!r}",
            f"elevation_m = {detector.elevation_m:g}",
            _arm_line("xarm_azimuth_rad", detector.xarm_azimuth_rad),
            _arm_line("yarm_azimuth_rad", detector.yarm_azimuth_rad),
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference_detector", help="Site code held fixed, e.g. S2.")
    parser.add_argument("detector", help="Site code to rotate, e.g. R2.")
    parser.add_argument(
        "--angle",
        type=float,
        required=True,
        help="Target relative orientation in degrees.",
    )
    parser.add_argument(
        "--reference",
        choices=get_args(OrientationReference),
        default="great_circle",
        help="Direction the orientations are measured from (default: great_circle).",
    )
    parser.add_argument(
        "--name",
        help="Table name of the printed block (default: the rotated detector's code).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rotated = align_detector(
        args.reference_detector,
        args.detector,
        math.radians(args.angle),
        args.reference,
    )
    print(geometry_block(rotated, args.name or args.detector))

    gamma = overlap_reduction_function(
        [SUMMARY_FREQUENCY_HZ], args.reference_detector, rotated
    )[0]
    for reference in get_args(OrientationReference):
        angle = relative_orientation(args.reference_detector, rotated, reference)
        print(f"{reference}: {math.degrees(angle):+.4f} deg", file=sys.stderr)
    print(f"ORF at {SUMMARY_FREQUENCY_HZ:g} Hz: {gamma:+.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
