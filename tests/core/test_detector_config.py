"""The core detector registry and its scientific identity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from astrogwb.detector.config import DetectorRegistry, load_detector_config
from astrogwb.simulators import detector_noise
from astrogwb.simulators.background import background_key

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
load_detector_config([Path({str(ROOT / "config/detectors.toml")!r})])
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


def test_background_identity_uses_only_referenced_scientific_settings() -> None:
    registry = _registry()
    key = background_key("waveforms", registry)
    detectors = dict(registry.detectors)
    detectors["E1"] = detectors["E1"].model_copy(update={"label": "presentation"})
    detectors["K1"] = detectors["K1"].model_copy(
        update={"psd_reference": "AplusDesign_psd.txt"}
    )
    changed_presentation = DetectorRegistry(
        detectors=detectors,
        networks={"pair": ("E2", "E1"), "triangle": ("E3", "E2", "E1")},
    )
    assert background_key("waveforms", changed_presentation) == key

    geometry = registry.detectors["E1"].geometry.model_copy(
        update={"longitude_rad": registry.detectors["E1"].geometry.longitude_rad + 0.1}
    )
    changed_geometry = DetectorRegistry(
        detectors={
            **registry.detectors,
            "E1": registry.detectors["E1"].model_copy(update={"geometry": geometry}),
        },
        networks=registry.networks,
    )
    assert background_key("waveforms", changed_geometry) != key


def test_geometry_overrides_change_noise_orfs_and_psd_overrides_change_noise() -> None:
    registry = _registry()
    frequencies = np.array([10.0, 20.0, 30.0])
    baseline = detector_noise(frequencies, registry, 1.0, network_names=["pair"])
    moved_geometry = registry.detectors["E1"].geometry.model_copy(
        update={"longitude_rad": registry.detectors["E1"].geometry.longitude_rad + 0.2}
    )
    geometry_registry = DetectorRegistry(
        detectors={
            **registry.detectors,
            "E1": registry.detectors["E1"].model_copy(
                update={"geometry": moved_geometry}
            ),
        },
        networks=registry.networks,
    )
    geometry_noise = detector_noise(
        frequencies, geometry_registry, 1.0, network_names=["pair"]
    )
    assert not np.array_equal(
        baseline.noise["pair"].pair_overlap,
        geometry_noise.noise["pair"].pair_overlap,
    )

    psd_registry = DetectorRegistry(
        detectors={
            **registry.detectors,
            "E1": registry.detectors["E1"].model_copy(
                update={"psd_reference": "ET_COBA_15km_psd.txt"}
            ),
        },
        networks=registry.networks,
    )
    psd_noise = detector_noise(frequencies, psd_registry, 1.0, network_names=["pair"])
    assert not np.array_equal(
        baseline.noise["pair"].effective_psd,
        psd_noise.noise["pair"].effective_psd,
    )
