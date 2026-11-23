"""Runtime detector registration and independently resolved network settings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import tomli_w

from astrogwb.detector import (
    Detector,
    load_detector,
    load_detectors_from_file,
    load_network,
    network_pairs,
    register_detector,
    register_network,
    select_network,
)


def test_loading_core_detector_config_does_not_initialize_xla(tmp_path: Path) -> None:
    path = tmp_path / "network.toml"
    names = ["L1", "H1"]
    detectors = {
        name: load_detector(name).model_dump(mode="json", exclude_none=True)
        for name in names
    }
    path.write_bytes(tomli_w.dumps({"detectors": detectors}).encode())
    code = """
import sys
from pathlib import Path
from astrogwb.detector import (
    load_detectors_from_file,
    load_network,
    register_network,
    select_network,
)

register_network("test-xla", {"H1", "L1"})
network = load_network("test-xla")
local = select_network(load_detectors_from_file(Path(sys.argv[1])), ["H1", "L1"])
for detector in (*network.values(), *local.values()):
    detector.geometry.location
    detector.geometry.response_tensor
import numpyro
numpyro.set_host_device_count(2)
import jax
assert jax.device_count() == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_network_pairs_sort_detector_keys() -> None:
    network = {name: load_detector(name) for name in ("E3", "E1", "E2")}
    assert network_pairs(network) == (
        ("E1", "E2"),
        ("E1", "E3"),
        ("E2", "E3"),
    )


def test_network_pairs_rejects_a_single_detector() -> None:
    with pytest.raises(ValueError, match="at least two detectors"):
        network_pairs({"H1": load_detector("H1")})


def test_network_pairs_rejects_a_key_that_differs_from_the_detector_name() -> None:
    network = {"H1": load_detector("L1"), "L1": load_detector("H1")}
    with pytest.raises(ValueError, match="keyed by"):
        network_pairs(network)


def test_registered_extensions_build_a_network_snapshot() -> None:
    suffix = uuid4().hex
    detector_name, network_name = f"X-{suffix}", f"network-{suffix}"
    detector = load_detector("L1").model_copy(
        update={"name": detector_name, "label": "Custom detector"}
    )
    members = {detector_name, "H1"}
    register_detector(detector)
    register_network(network_name, members)
    members.clear()

    runtime = load_detector(detector_name)
    assert runtime is detector
    assert runtime.label == "Custom detector"

    snapshot = load_network(network_name)
    loaded = load_network(network_name)
    assert snapshot is not loaded
    assert tuple(snapshot) == ("H1", detector_name)
    assert snapshot[detector_name] is loaded[detector_name] is detector
    assert snapshot["H1"] is loaded["H1"] is load_detector("H1")
    with pytest.raises(ValueError, match="already registered"):
        register_detector(detector)
    with pytest.raises(ValueError, match="already registered"):
        register_network(network_name, {"H1"})


def test_toml_and_process_catalog_are_isolated(tmp_path: Path) -> None:
    suffix = uuid4().hex
    name, extension = f"network-{suffix}", f"X-{suffix}"
    catalog_s1 = load_detector("S1")
    register_detector(load_detector("H1").model_copy(update={"name": extension}))
    register_network(name, {"S1", extension})
    path = tmp_path / "network.toml"
    detectors = {
        detector: load_detector(detector).model_dump(mode="json", exclude_none=True)
        for detector in ("S1", "C1")
    }
    detectors["S1"]["geometry"]["xarm_azimuth_rad"] = 0.5
    path.write_bytes(tomli_w.dumps({"detectors": detectors}).encode())

    local = select_network(load_detectors_from_file(path), ["S1", "C1"])
    assert network_pairs(local) == (("C1", "S1"),)
    assert local["S1"].geometry.xarm_azimuth_rad == 0.5
    snapshot = load_network(name)
    assert network_pairs(snapshot) == (("S1", extension),)
    assert snapshot["S1"].geometry == catalog_s1.geometry

    with pytest.raises(ValueError, match=f"unknown detectors.*{extension}"):
        select_network(load_detectors_from_file(path), ["S1", extension])


def test_select_network_returns_sorted_detectors() -> None:
    detectors = {name: load_detector(name) for name in ("L1", "H1", "V1")}

    network = select_network(detectors, ["V1", "H1"])

    assert tuple(network) == ("H1", "V1")
    assert network["H1"] is detectors["H1"]


def test_select_network_rejects_unknown_detectors() -> None:
    detectors = {name: load_detector(name) for name in ("H1", "L1")}
    with pytest.raises(ValueError, match=r"unknown detectors.*X9.*Available.*H1, L1"):
        select_network(detectors, ["H1", "X9"])


def test_select_network_rejects_duplicate_detectors() -> None:
    detectors = {name: load_detector(name) for name in ("H1", "L1")}
    with pytest.raises(ValueError, match="duplicate detector names"):
        select_network(detectors, ["H1", "H1"])


def test_standalone_detector_dump_and_lazy_psd_copy(tmp_path: Path) -> None:
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    np.savetxt(first, [[20.0, 1.0], [40.0, 2.0]])
    np.savetxt(second, [[20.0, 3.0], [40.0, 4.0]])
    detector = Detector(
        name="custom",
        geometry=load_detector("S1").geometry,
        psd_reference=str(first),
        label="Custom detector",
    )
    before = detector.model_dump(mode="json")
    assert before["name"] == "custom"
    assert before["label"] == "Custom detector"
    assert detector.geometry.location.shape == (3,)
    assert detector.geometry.response_tensor.shape == (3, 3)
    sensitivity = detector.sensitivity
    np.savetxt(first, [[20.0, 5.0], [40.0, 6.0]])
    np.testing.assert_array_equal(sensitivity([20.0, 40.0]), [5.0, 6.0])
    assert detector.model_dump(mode="json") == before
    assert Detector.model_validate_json(detector.model_dump_json()) == detector
    assert Detector.model_validate(before).sensitivity is sensitivity

    updated = detector.model_copy(update={"psd_reference": str(second)})
    np.testing.assert_array_equal(updated.sensitivity([20.0, 40.0]), [3.0, 4.0])
