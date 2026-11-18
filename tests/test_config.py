"""Detector network loading from local TOML layers."""

from pathlib import Path

import pytest
from astrogwb_tiny_pipeline import packaged_config
from pydantic import ValidationError

from astrogwb.detector import DetectorNetworkConfig, load_network


def test_detector_loader_merges_packaged_definitions_and_ordered_overrides(
    tmp_path: Path,
) -> None:
    packaged_s1 = packaged_config("S1")
    base = tmp_path / "networks.toml"
    base.write_text(
        '[networks]\nreference = ["S1", "R1"]\n'
        'comparison = "${networks.reference}"\n'
        'with_custom = ["S1", "custom"]\n'
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
    reference = DetectorNetworkConfig.from_files(base, override, network="reference")
    comparison = DetectorNetworkConfig.from_files(base, override, network="comparison")
    assert reference.pairs() == (("C1", "S1"),)
    assert set(reference.detectors) == {"C1", "S1"}
    assert comparison.detectors == reference.detectors

    local = DetectorNetworkConfig.from_files(base, override, network="with_custom")
    assert set(local.detectors) == {"S1", "custom"}
    custom = local.detectors["custom"]
    assert custom.geometry.xarm_azimuth_rad == 0.5
    assert custom.geometry.latitude_rad == packaged_s1.geometry.latitude_rad
    assert custom.psd_reference == packaged_config("C1").psd_reference
    assert custom.label == "Custom"

    original = DetectorNetworkConfig.from_files(base, network="reference")
    assert original.pairs() == (("R1", "S1"),)
    with pytest.raises(KeyError, match="with_custom"):
        load_network("with_custom")


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("[spectra]\nnum_draws = 5\n", "Extra inputs are not permitted"),
        ('[networks]\ninvalid = ["S1", "S1"]\n', "distinct detectors"),
        ('[networks]\ninvalid = ["S1", "missing"]\n', "undefined detectors.*missing"),
        (
            (
                '[networks]\ninvalid = ["S1", "C1"]\n'
                '[detectors.S1]\npsd_reference = "missing-curve.txt"\n'
            ),
            "missing-curve",
        ),
    ],
    ids=["extra-table", "duplicate-member", "unknown-member", "invalid-psd"],
)
def test_detector_loader_rejects_invalid_toml_layers(
    tmp_path: Path, document: str, message: str
) -> None:
    layer = tmp_path / "invalid.toml"
    layer.write_text(document, encoding="utf-8")
    with pytest.raises(ValidationError, match=message):
        DetectorNetworkConfig.from_files(layer, network="invalid")
