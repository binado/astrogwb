"""The core detector registry and its scientific identity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from astrogwb.detector.config import DetectorRegistry, load_detector_config

ROOT = Path(__file__).parents[2]


def _registry() -> DetectorRegistry:
    base = load_detector_config([ROOT / "config/detectors.toml"])
    return DetectorRegistry(
        detectors=base.detectors,
        networks={
            "triangle": ("E1", "E2", "E3"),
            "pair": ("E1", "E2"),
        },
    )


def test_loading_core_detector_config_does_not_initialize_xla() -> None:
    code = f"""
from pathlib import Path
from astrogwb.detector.config import load_detector_config
registry = load_detector_config([Path({str(ROOT / "config/detectors.toml")!r})])
for detector in registry.build_detectors(["H1", "L1"]):
    detector.geometry.location
    detector.geometry.response_tensor
import numpyro
numpyro.set_host_device_count(2)
import jax
assert jax.device_count() == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_pairs_are_canonical_for_reordered_and_overlapping_networks() -> None:
    registry = _registry()
    assert registry.pairs("triangle") == [
        ("E1", "E2"),
        ("E1", "E3"),
        ("E2", "E3"),
    ]
    assert registry.pairs("pair") == [("E1", "E2")]


def test_owned_registry_build_order_and_serialization() -> None:
    registry = _registry()
    before = registry.model_dump(mode="json")
    names = ("L1", "H1", "E1")
    built = registry.build_detectors(names)
    assert tuple(detector.name for detector in built) == names
    assert tuple(detector.name for detector in registry.build_network("triangle")) == (
        "E1",
        "E2",
        "E3",
    )
    for detector in built:
        assert detector.geometry is registry.detectors[detector.name].geometry
        assert detector.geometry.location.shape == (3,)
        assert detector.geometry.response_tensor.shape == (3, 3)
    assert registry.model_dump(mode="json") == before
