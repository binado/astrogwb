"""Detector-noise metadata format and registry-backed reads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrogwb.detector.config import DetectorRegistry, load_detector_config
from astrogwb.simulators import DetectorNoise, DetectorNoiseMetadata, detector_noise
from astrogwb.utils.io import write

ROOT = Path(__file__).parents[2]


def _registry() -> DetectorRegistry:
    base = load_detector_config([ROOT / "config/detectors.toml"])
    return DetectorRegistry(
        detectors=base.detectors,
        networks={"pair": ("R1", "S1")},
    )


def test_noise_metadata_stores_network_pairs_and_round_trips(
    tmp_path: Path,
) -> None:
    registry = _registry()
    noise = detector_noise(
        np.array([10.0, 20.0, 30.0]), registry, 1.0, network_names=["pair"]
    )
    path = noise.write(tmp_path / "noise.h5")
    recorded = DetectorNoise.open(path)
    assert recorded.metadata.pairs == {"pair": (("R1", "S1"),)}
    assert "detectors" not in recorded.metadata.model_dump()
    np.testing.assert_array_equal(
        recorded.network("pair").pair_overlap, noise.network("pair").pair_overlap
    )


def test_old_noise_metadata_requires_regeneration(tmp_path: Path) -> None:
    path = tmp_path / "old-noise.h5"
    metadata = DetectorNoiseMetadata(
        format_version=1,
        pairs={"pair": (("R1", "S1"),)},
        scientific_identity="old",
        observation_time=1.0,
    )
    pair = np.zeros((1, 2))
    frequency = np.zeros(2)
    write(
        path,
        {
            "frequencies": frequency,
            "networks": {
                "pair": {
                    "pair_overlap": pair,
                    "pair_scale": pair,
                    "effective_psd": frequency,
                    "scale": frequency,
                }
            },
        },
        metadata,
    )
    with pytest.raises(ValueError, match="regenerate"):
        DetectorNoise.open(path)
