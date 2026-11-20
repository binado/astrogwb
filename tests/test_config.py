"""Detector network loading from local TOML layers."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from astrogwb.detector import (
    load_detector,
    load_network,
    load_networks_from_files,
    network_pairs,
)


def test_detector_loader_merges_packaged_definitions_and_ordered_overrides(
    tmp_path: Path,
) -> None:
    packaged_s1 = load_detector("S1")
    base = tmp_path / "networks.toml"
    base.write_text(
        '[networks]\nreference = ["S1", "R1"]\n'
        'comparison = "${networks.reference}"\n'
        'with_custom = ["S1", "custom"]\n'
        '[detectors]\ncustom = "${detectors.S1}"\n',
        encoding="utf-8",
    )
    override = tmp_path / "override.toml"
    override.write_text(
        '[networks]\nreference = ["S1", "C1"]\n'
        "[detectors.custom.geometry]\nxarm_azimuth_rad = 0.5\n"
        '[detectors.custom]\npsd_reference = "${detectors.C1.psd_reference}"\n'
        'label = "Custom"\n',
        encoding="utf-8",
    )
    networks = load_networks_from_files(base, override)
    assert tuple(networks) == ("comparison", "reference", "with_custom")
    reference = networks["reference"]
    assert tuple(reference) == ("C1", "S1")
    assert networks["comparison"]["C1"] is reference["C1"]

    local = networks["with_custom"]
    assert tuple(local) == ("S1", "custom")
    assert local["S1"] is reference["S1"]
    custom = local["custom"]
    assert custom.geometry.xarm_azimuth_rad == 0.5
    assert custom.geometry.latitude_rad == packaged_s1.geometry.latitude_rad
    assert custom.psd_reference == load_detector("C1").psd_reference
    assert custom.name == "custom"
    assert custom.label == "Custom"

    original = load_networks_from_files(base)["reference"]
    assert network_pairs(original) == (("R1", "S1"),)
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
        (
            '[detectors.S1]\nname = "C1"\n',
            "detector name 'C1'.*mapping key 'S1'",
        ),
    ],
    ids=[
        "extra-table",
        "duplicate-member",
        "unknown-member",
        "invalid-psd",
        "mismatched-name",
    ],
)
def test_detector_loader_rejects_invalid_toml_layers(
    tmp_path: Path, document: str, message: str
) -> None:
    layer = tmp_path / "invalid.toml"
    layer.write_text(document, encoding="utf-8")
    with pytest.raises(ValidationError, match=message):
        load_networks_from_files(layer)
