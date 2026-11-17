"""Generic detector configuration layering."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from astrogwb.detector.config import load_detector_config


def test_detector_loader_merges_packaged_definitions_and_ordered_overrides(
    tmp_path: Path,
) -> None:
    packaged = load_detector_config([])
    assert packaged.networks == {}
    base = tmp_path / "networks.toml"
    base.write_text(
        '[networks]\nreference = ["S1", "R1"]\n'
        'comparison = "${networks.reference}"\n'
        '[detectors.custom]\nextends = "${detectors.S1}"\nlabel = "Custom"\n',
        encoding="utf-8",
    )
    override = tmp_path / "override.toml"
    override.write_text(
        '[networks]\nreference = ["S1", "C1"]\n'
        "[detectors.custom.geometry]\nxarm_azimuth_rad = 0.5\n"
        '[detectors.custom]\npsd_reference = "${detectors.C1.psd_reference}"\n',
        encoding="utf-8",
    )
    registry = load_detector_config([base, override])
    assert registry.networks["reference"] == ("S1", "C1")
    assert registry.networks["comparison"] == ("S1", "C1")
    assert registry.pairs("reference") == [("C1", "S1")]
    assert registry.detectors["custom"].geometry.xarm_azimuth_rad == 0.5
    assert (
        registry.detectors["custom"].geometry.latitude_rad
        == packaged.detectors["S1"].geometry.latitude_rad
    )
    assert (
        registry.detectors["custom"].psd_reference
        == packaged.detectors["C1"].psd_reference
    )
    assert registry.detectors["custom"].label == "Custom"


def test_detector_loader_rejects_spectrum_settings(tmp_path: Path) -> None:
    layer = tmp_path / "wrong-group.toml"
    layer.write_text("[spectra]\nnum_draws = 5\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_detector_config([layer])
