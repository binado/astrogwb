"""Tests for asgwb.detector module (PowerSpectralDensity and Detector)."""

from __future__ import annotations

from pathlib import Path

import pytest

from asgwb.detector.detector import Detector
from asgwb.detector.psd import PowerSpectralDensity


@pytest.fixture
def h1_dict() -> dict[str, float | str]:
    return {
        "name": "H1",
        "minimum_frequency": 20.0,
        "maximum_frequency": 2048.0,
        "length": 4.0,
        "latitude": 46.45514666666667,
        "longitude": -119.4076571388889,
        "elevation": 142.554,
        "xarm_azimuth": 125.9994,
        "yarm_azimuth": 215.9994,
        "xarm_tilt": -0.0006195,
        "yarm_tilt": 1.25e-05,
        "duty_factor": 0.7,
    }


@pytest.fixture
def aplus_psd() -> PowerSpectralDensity:
    return PowerSpectralDensity.from_noise_curve_dir("AplusDesign_psd.txt")


def test_from_noise_curve_dir_valid():
    psd = PowerSpectralDensity.from_noise_curve_dir("AplusDesign_psd.txt")
    assert psd.file.name == "AplusDesign_psd.txt"
    assert psd.file.exists()
    assert psd.curve_type == "psd"


def test_from_noise_curve_dir_missing():
    with pytest.raises(FileNotFoundError):
        PowerSpectralDensity.from_noise_curve_dir("nonexistent_noise_curve.txt")


def test_to_bilby_psd_unknown_curve_type():
    # curve_type validation now happens before any bilby import, so this is
    # bilby-free and must raise ValueError regardless of bilby availability.
    psd = PowerSpectralDensity(file=Path("irrelevant.txt"), curve_type="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown curve type"):
        psd.to_bilby_psd()


def test_from_dict(h1_dict: dict[str, float | str], aplus_psd: PowerSpectralDensity):
    det = Detector.from_dict(h1_dict, psd=aplus_psd)

    assert det.name == "H1"
    assert det.psd is aplus_psd
    assert det.length == 4.0
    assert det.minimum_frequency == 20.0
    assert det.maximum_frequency == 2048.0
    assert det.duty_factor == 0.7


def test_from_dict_with_string_psd(h1_dict: dict[str, float | str]):
    det = Detector.from_dict(h1_dict, psd="AplusDesign_psd.txt")

    assert isinstance(det.psd, PowerSpectralDensity)
    assert det.psd.file.name == "AplusDesign_psd.txt"


def test_from_dict_no_psd_raises(h1_dict: dict[str, float | str]):
    # dict has no default_noise_curve key and psd arg is None → KeyError
    incomplete = {k: v for k, v in h1_dict.items() if k != "name"}
    with pytest.raises(KeyError):
        Detector.from_dict(incomplete, psd=None)


def test_from_file_h1():
    det = Detector.from_file("H1")

    assert det.name == "H1"
    assert det.length == 4.0
    assert det.minimum_frequency == 20.0
    assert det.maximum_frequency == 2048.0
    assert isinstance(det.psd, PowerSpectralDensity)


def test_from_file_missing_detector():
    with pytest.raises(KeyError):
        Detector.from_file("NONEXISTENT")


def test_from_file_custom_path(tmp_path: Path):
    toml_content = """\
[mytable.TEST]
default_noise_curve = "AplusDesign_psd.txt"
minimum_frequency = 10.0
maximum_frequency = 1024.0
length = 2.0
latitude = 0.0
longitude = 0.0
elevation = 0.0
xarm_azimuth = 0.0
yarm_azimuth = 90.0
xarm_tilt = 0.0
yarm_tilt = 0.0
duty_factor = 0.8
"""
    toml_file = tmp_path / "custom.toml"
    toml_file.write_text(toml_content)

    det = Detector.from_file("TEST", path=toml_file, table="mytable")

    assert det.name == "TEST"
    assert det.length == 2.0
    assert det.duty_factor == 0.8


def test_to_bilby_psd(aplus_psd: PowerSpectralDensity):
    import bilby

    bilby_psd = aplus_psd.to_bilby_psd()
    assert isinstance(bilby_psd, bilby.gw.detector.PowerSpectralDensity)


def test_to_bilby_detector():
    import bilby

    det = Detector.from_file("H1")
    ifo = det.to_bilby_detector()
    assert isinstance(ifo, bilby.gw.detector.Interferometer)
    assert ifo.name == "H1"
    assert ifo.length == 4.0
