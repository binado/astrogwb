"""Detector loading from complete TOML documents."""

import tomllib
from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

from astrogwb.detector import (
    load_detector,
    load_detectors_from_file,
)


def test_detector_loader_reads_every_detector_from_one_file(tmp_path: Path) -> None:
    packaged_s1 = load_detector("S1")
    packaged_c1 = load_detector("C1")
    s1 = packaged_s1.model_dump(mode="json", exclude={"name"}, exclude_none=True)
    c1 = packaged_c1.model_dump(mode="json", exclude={"name"}, exclude_none=True)
    custom = {
        "geometry": {**s1["geometry"], "xarm_azimuth_rad": 0.5},
        "psd_reference": c1["psd_reference"],
        "label": "Custom",
    }
    path = tmp_path / "detectors.toml"
    path.write_bytes(
        tomli_w.dumps({"detectors": {"S1": s1, "C1": c1, "custom": custom}}).encode()
    )

    detectors = load_detectors_from_file(path)

    assert tuple(detectors) == ("S1", "C1", "custom")
    custom = detectors["custom"]
    assert custom.geometry.xarm_azimuth_rad == 0.5
    assert custom.geometry.latitude_rad == packaged_s1.geometry.latitude_rad
    assert custom.psd_reference == packaged_c1.psd_reference
    assert custom.name == "custom"
    assert custom.label == "Custom"


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("[spectra]\nnum_draws = 5\n", "Extra inputs are not permitted"),
        ('[networks]\nHL = ["S1", "C1"]\n', "Extra inputs are not permitted"),
        ('[detectors.S1]\npsd_reference = "missing-curve.txt"\n', "missing-curve"),
        (
            '[detectors.S1]\nname = "C1"\n',
            "detector name 'C1'.*mapping key 'S1'",
        ),
    ],
    ids=[
        "extra-table",
        "networks-table",
        "invalid-psd",
        "mismatched-name",
    ],
)
def test_detector_loader_rejects_invalid_documents(
    tmp_path: Path, document: str, message: str
) -> None:
    path = tmp_path / "invalid.toml"
    settings = tomllib.loads(document)
    detectors = settings.setdefault("detectors", {})
    for name in ("S1", "C1"):
        fields = load_detector(name).model_dump(
            mode="json", exclude={"name"}, exclude_none=True
        )
        detectors[name] = fields | detectors.get(name, {})
    path.write_bytes(tomli_w.dumps(settings).encode())
    with pytest.raises(ValidationError, match=message):
        load_detectors_from_file(path)
