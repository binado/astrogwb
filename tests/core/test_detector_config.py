"""The core detector registry and its scientific identity."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from astrogwb.detector.config import DetectorRegistry, load_detector_config
from astrogwb.simulators import detector_noise
from astrogwb.simulators.background import background_key, scientific_payload

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

    geometry = dataclasses.replace(
        registry.detectors["E1"].geometry,
        longitude_rad=registry.detectors["E1"].geometry.longitude_rad + 0.1,
    )
    changed_geometry = DetectorRegistry(
        detectors={
            **registry.detectors,
            "E1": registry.detectors["E1"].model_copy(update={"geometry": geometry}),
        },
        networks=registry.networks,
    )
    assert background_key("waveforms", changed_geometry) != key


@pytest.mark.parametrize(("tilt", "recorded"), [(0.0, False), (1e-3, True)])
def test_payload_records_an_arm_tilt_only_when_it_is_not_the_default(
    tilt: float, recorded: bool
) -> None:
    registry = _registry()
    geometry = dataclasses.replace(
        registry.detectors["E1"].geometry, xarm_tilt_rad=tilt
    )
    tilted = DetectorRegistry(
        detectors={
            **registry.detectors,
            "E1": registry.detectors["E1"].model_copy(update={"geometry": geometry}),
        },
        networks=registry.networks,
    )

    fields = scientific_payload(tilted, ["pair"])["detectors"]["E1"]["geometry"]

    assert ("xarm_tilt_rad" in fields) is recorded


def test_geometry_overrides_change_noise_orfs_and_psd_overrides_change_noise() -> None:
    registry = _registry()
    frequencies = np.array([10.0, 20.0, 30.0])
    baseline = detector_noise(frequencies, registry, 1.0, network_names=["pair"])
    moved_geometry = dataclasses.replace(
        registry.detectors["E1"].geometry,
        longitude_rad=registry.detectors["E1"].geometry.longitude_rad + 0.2,
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
