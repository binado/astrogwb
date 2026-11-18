"""Runtime detector registration and independently resolved network settings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from astrogwb_tiny_pipeline import packaged_config

from astrogwb.detector import (
    DetectorNetworkConfig,
    load_detector,
    load_network,
    register_detector,
    register_network,
)


def test_loading_core_detector_config_does_not_initialize_xla() -> None:
    code = """
from astrogwb.detector import DetectorNetworkConfig, load_network, register_network

register_network("test-xla", {"H1", "L1"})
network = DetectorNetworkConfig.from_network("test-xla")
for detector in (*network.build_detectors(), *load_network(network.name)):
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


def test_network_build_order_pairs_and_serialization_are_canonical() -> None:
    network = DetectorNetworkConfig(
        name="triangle",
        detectors={name: packaged_config(name) for name in ("E3", "E1", "E2")},
    )
    before = network.model_dump(mode="json")
    assert network.pairs() == (
        ("E1", "E2"),
        ("E1", "E3"),
        ("E2", "E3"),
    )
    built = network.build_detectors()
    assert tuple(detector.name for detector in built) == ("E1", "E2", "E3")
    for detector in built:
        assert detector.geometry is network.detectors[detector.name].geometry
    assert network.model_dump(mode="json") == before
    assert (
        DetectorNetworkConfig.model_validate_json(network.model_dump_json()) == network
    )


def test_registered_extensions_build_a_network_snapshot() -> None:
    suffix = uuid4().hex
    detector_name, network_name = f"X-{suffix}", f"network-{suffix}"
    config = packaged_config("L1")
    members = {detector_name, "H1"}
    register_detector(detector_name, config)
    register_network(network_name, members)
    members.clear()

    runtime = load_detector(detector_name)
    assert runtime.name == detector_name
    assert runtime.geometry == config.geometry
    assert runtime.sensitivity.psd_reference == config.psd_reference

    snapshot = DetectorNetworkConfig.from_network(network_name)
    assert snapshot.name == network_name
    assert snapshot.detectors == {"H1": packaged_config("H1"), detector_name: config}
    assert snapshot.pairs() == (("H1", detector_name),)
    assert load_network(network_name) == snapshot.build_detectors()
    with pytest.raises(ValueError, match="already registered"):
        register_detector(detector_name, config)
    with pytest.raises(ValueError, match="already registered"):
        register_network(network_name, {"H1"})


def test_toml_and_process_catalog_are_isolated(tmp_path: Path) -> None:
    suffix = uuid4().hex
    name, extension = f"network-{suffix}", f"X-{suffix}"
    catalog_s1 = load_detector("S1")
    register_detector(extension, packaged_config("H1"))
    register_network(name, {"S1", extension})
    layer = tmp_path / "network.toml"
    layer.write_text(
        f'[networks]\n{name} = ["S1", "C1"]\n'
        "[detectors.S1.geometry]\nxarm_azimuth_rad = 0.5\n",
        encoding="utf-8",
    )

    local = DetectorNetworkConfig.from_files(layer, network=name)
    assert local.pairs() == (("C1", "S1"),)
    assert local.detectors["S1"].geometry.xarm_azimuth_rad == 0.5
    snapshot = DetectorNetworkConfig.from_network(name)
    assert snapshot.pairs() == (("S1", extension),)
    assert snapshot.detectors["S1"].geometry == catalog_s1.geometry

    layer.write_text(f'[networks]\n{name} = ["S1", "{extension}"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=f"undefined detectors.*{extension}"):
        DetectorNetworkConfig.from_files(layer, network=name)
