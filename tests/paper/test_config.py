"""Tests for shared configuration loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from repo import REPO_ROOT

from astrogwb.config import load_config
from astrogwb.detector.config import load_detector_config

PAPER_ROOT = REPO_ROOT


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
    # Members keep the order of the last layer; pairs are in canonical order.
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


def test_load_config_shared_values_and_references() -> None:
    config = load_config(REPO_ROOT / "config")
    assert config.fiducials["H0"] == 67.66
    assert config.catalog.fiducials == config.fiducials
    assert config.catalog.waveform == config.waveforms["default"]
    assert (
        config.waveforms["TaylorF2"].minimum_frequency
        == config.waveforms["default"].minimum_frequency
    )
    assert config.registry.networks["ET-2L-aligned"] == ("S1", "R1")
    assert "S1" in config.registry.detectors
    assert config.figure_format == "pdf"
    assert "H0" in config.labels
    assert config.priors["H0"].build().__class__.__name__ == "Uniform"


def test_load_config_overrides_and_independent_instances(tmp_path: Path) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first.write_text(
        '[fiducials]\nH0 = 70.0\n[priors]\nH0 = {dist = "Normal", kwargs = {loc = 70.0, scale = 2.0}}\n[detectors.H1]\npsd_referenc = "ignored"\n'
    )
    second.write_text("[fiducials]\nH0 = 72.0\n")
    config = load_config(REPO_ROOT / "config", overrides=(first, second))
    assert config.fiducials["H0"] == 72.0
    assert config.catalog.fiducials["H0"] == 72.0
    assert config.priors["H0"].kwargs == {"loc": 70.0, "scale": 2.0}
    config.fiducials["H0"] = 999.0
    assert load_config(REPO_ROOT / "config").fiducials["H0"] == 67.66


def test_load_config_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no config TOML files"):
        load_config(tmp_path)


def test_waveform_variants_and_validated_overrides() -> None:
    from astrogwb.waveform import WaveformMetadata, waveform_generator

    config = load_config(REPO_ROOT / "config")
    default = config.catalog.waveform
    generator = waveform_generator(default)
    assert np.asarray(generator.frequencies).size == 400
    assert config.waveforms["linear"].frequency_spacing == "linear"
    assert config.waveforms["TaylorF2"].approximant == "TaylorF2"
    modified = WaveformMetadata.model_validate(
        {**default.model_dump(), "approximant": "AnalyticInspiral", "alpha": 0.02}
    )
    assert modified.alpha == 0.02
    with pytest.raises(ValidationError, match="alpha is only valid"):
        WaveformMetadata.model_validate({**default.model_dump(), "alpha": 0.02})
