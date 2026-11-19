"""Detector geometry against bilby's bundled interferometers."""

from __future__ import annotations

import bilby
import pytest
from tolerances import assert_close

from astrogwb.detector import load_detector

SITES = ["H1", "L1", "V1", "K1"]


@pytest.fixture(params=SITES)
def site(request: pytest.FixtureRequest) -> str:
    return request.param


def test_vertex_matches_bilby_within_a_millimetre(site: str) -> None:
    reference = bilby.gw.detector.get_empty_interferometer(site)

    location = load_detector(site).geometry.location

    assert_close(location, reference.vertex)


def test_response_tensor_matches_bilby(site: str) -> None:
    """Sensitive to arm azimuths and tilts: both are radians-scale entries."""
    reference = bilby.gw.detector.get_empty_interferometer(site)

    tensor = load_detector(site).geometry.response_tensor

    assert_close(tensor, reference.detector_tensor)
