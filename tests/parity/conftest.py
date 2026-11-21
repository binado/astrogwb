"""Shared fixtures of the parity suite.

Each module here compares one interface of the library against an established
package. bilby is called directly. pygwb cannot share this environment, so its
values are read from ``references/pygwb.npz``, written by
``references/generate_pygwb.py``; the archive also carries every input both
sides read, so a test never restates a pygwb formula.

The two packages normalize the strain spectrum differently. pygwb's
``S_alpha`` is the cross power of co-located aligned detectors,
:math:`3 H_0^2 \\Omega / (10 \\pi^2 f^3)`, which is :data:`PYGWB_POWER_FRACTION`
of this library's :math:`S_h`; the library keeps that factor next to the
overlap reduction function instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np
import pytest
from numpy.typing import NDArray

# Module scope, above the library imports: must precede any array creation.
jax.config.update("jax_enable_x64", True)

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import (
    Detector,
    DetectorGeometry,
    Network,
)

REFERENCES = Path(__file__).parent / "references"

#: pygwb's strain power per unit of this library's spectral density.
PYGWB_POWER_FRACTION = 2.0 / 5.0


def geometry_from_bilby_convention(
    latitude: float,
    longitude: float,
    elevation: float,
    xarm_azimuth: float,
    yarm_azimuth: float,
) -> DetectorGeometry:
    """Geometry of a site given in degrees with azimuths counter-clockwise from East.

    Parameters
    ----------
    latitude, longitude
        Geodetic coordinates in degrees.
    elevation
        Elevation in metres.
    xarm_azimuth, yarm_azimuth
        Arm azimuths in degrees, counter-clockwise from East.
    """

    def from_north(azimuth: float) -> float:
        return (math.pi / 2.0 - math.radians(azimuth)) % (2.0 * math.pi)

    return DetectorGeometry(
        latitude_rad=math.radians(latitude),
        longitude_rad=math.radians(longitude),
        elevation_m=elevation,
        xarm_azimuth_rad=from_north(xarm_azimuth),
        yarm_azimuth_rad=from_north(yarm_azimuth),
    )


@pytest.fixture(scope="session")
def pygwb_reference() -> Mapping[str, NDArray[np.float64]]:
    """The frozen pygwb archive, loaded once per worker."""
    with np.load(REFERENCES / "pygwb.npz") as archive:
        return {name: archive[name] for name in archive.files}


@pytest.fixture(scope="session")
def hubble_constant(pygwb_reference: Mapping[str, NDArray[np.float64]]) -> float:
    """The Hubble constant pygwb ran with, in km/s/Mpc."""
    return float(pygwb_reference["hubble_constant"])


@pytest.fixture(scope="session")
def frequencies(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """The uniform grid of the ORF, estimator and likelihood sections, in Hz."""
    return pygwb_reference["frequencies"]


@pytest.fixture(scope="session")
def observation_years(pygwb_reference: Mapping[str, NDArray[np.float64]]) -> float:
    """The estimator's segment duration, in the years the library takes."""
    return float(pygwb_reference["segment_duration"]) / SECONDS_PER_YEAR


@pytest.fixture(scope="session")
def exact_network(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    tmp_path_factory: pytest.TempPathFactory,
) -> Network:
    """The archive's exact sites with its noise curves, as one network.

    The curves are written on the archive's own grid, so the library
    interpolates them at their nodes.
    """
    directory = tmp_path_factory.mktemp("noise_curves")
    names = [str(name) for name in pygwb_reference["site_exact_names"]]
    detectors = {}
    coordinates = pygwb_reference["site_exact_coordinates"].tolist()
    for index, name in enumerate(names):
        path = directory / f"{name}_psd.txt"
        curve = [pygwb_reference["frequencies"], pygwb_reference["psd"][index]]
        np.savetxt(path, np.column_stack(curve))
        detectors[name] = Detector(
            name=name,
            geometry=geometry_from_bilby_convention(*coordinates[index]),
            psd_reference=str(path),
        )
    return detectors


@pytest.fixture(scope="session")
def exact_detectors(exact_network: Network) -> tuple[Detector, ...]:
    """The exact network's detectors, in the pair order of the archive."""
    return tuple(exact_network[name] for name in sorted(exact_network))
