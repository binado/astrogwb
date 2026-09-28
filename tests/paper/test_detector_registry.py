"""Detector definitions are shared, serializable, and specific to each run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from config_fixtures import example_raw, write_defaults
from pydantic import ValidationError
from repo import REPO_ROOT

from astrogwb.paper.config import detector_registry, mcmc
from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.config.runs import assemble_run, resolve_networks


def test_defaults_reproduce_packaged_geometry_and_effective_psd() -> None:
    from astrogwb.detector import effective_psd, load_detector, load_sensitivity_map

    registry = detector_registry()
    frequencies = np.array([5.0, 10.0, 50.0, 100.0, 500.0])
    for name, members in registry.networks.items():
        geometry, sensitivities = registry.build_network(name)
        assert [detector.name for detector in geometry] == list(members)
        for detector in geometry:
            assert detector == load_detector(detector.name)
            assert registry.detectors[detector.name].label == detector.name
        np.testing.assert_array_equal(
            effective_psd(frequencies, geometry, sensitivities),
            effective_psd(frequencies, members, load_sensitivity_map(members)),
        )


def test_partial_geometry_and_psd_overrides_preserve_other_fields() -> None:
    default = detector_registry()
    registry = detector_registry(
        detectors={
            "E1": {
                "geometry": {"xarm_azimuth_rad": 0.3},
                "psd_reference": "ET_D_psd",
                "label": "ET channel 1",
            }
        }
    )
    first = registry.detectors["E1"]
    assert first.geometry.latitude_rad == default.detectors["E1"].geometry.latitude_rad
    assert first.geometry.xarm_azimuth_rad == 0.3
    assert first.label == "ET channel 1"
    geometry, sensitivities = registry.build_network("ET-triangular")
    assert geometry[0].xarm_azimuth_rad == 0.3
    assert sensitivities["E1"].psd_reference == "ET_D_psd"


def test_new_detector_requires_complete_definition_and_defaults_tilts() -> None:
    registry = detector_registry(
        detectors={
            "new": {
                "geometry": {
                    "latitude_rad": 0.1,
                    "longitude_rad": 0.2,
                    "elevation_m": 30.0,
                    "xarm_azimuth_rad": 0.4,
                    "yarm_azimuth_rad": 1.3,
                },
                "psd_reference": "ET_D_psd",
            }
        },
        networks={"custom": ["new", "E1"]},
    )
    geometry, sensitivities = registry.build_network("custom")
    assert [detector.name for detector in geometry] == ["new", "E1"]
    assert geometry[0].xarm_tilt_rad == geometry[0].yarm_tilt_rad == 0.0
    assert set(sensitivities) == {"new", "E1"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"new": {"psd_reference": "ET_D_psd"}},
        {"new": {"geometry": {"latitude_rad": 0.1}}},
        {"E1": {"psd_referenc": "ET_D_psd"}},
        {"E1": {"geometry": {"azimuth": 10.0}}},
        {"E1": {"geometry": {"latitude_rad": float("nan")}}},
        {"E1": {"psd_reference": ""}},
        {"E1": {"psd_reference": "does-not-exist.txt"}},
    ],
)
def test_invalid_definitions_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        detector_registry(detectors=overrides)


@pytest.mark.parametrize("members", [[], ["E1", "unknown"], ["E1", "E1"]])
def test_invalid_members_are_rejected(members: list[str]) -> None:
    with pytest.raises(ValidationError):
        detector_registry(networks={"bad": members})


def test_registry_instances_are_independent() -> None:
    first = detector_registry()
    second = detector_registry()
    first.detectors.pop("E1")
    first.networks.clear()
    assert "E1" in second.detectors
    assert second.networks
    assert detector_registry() == second


def test_save_reload_uses_resolved_settings_without_packaged_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_run_config(
        example_raw(), detectors={"S1": {"geometry": {"elevation_m": 999.0}}}
    )
    path = tmp_path / "run.json"
    config.save(path)
    saved = json.loads(path.read_text())
    monkeypatch.setattr(
        mcmc.knf,
        "load",
        lambda *_args, **_kwargs: pytest.fail("reinterpreted saved settings"),
    )
    reloaded = build_run_config(saved)
    assert reloaded.model_dump(mode="json") == config.model_dump(mode="json")
    assert reloaded.detector_registry.detectors["S1"].geometry.elevation_m == 999.0


def test_legacy_saved_config_gets_packaged_registry() -> None:
    config = build_run_config(example_raw())
    saved = config.model_dump(mode="json")
    saved.pop("detector_registry")
    reloaded = build_run_config(saved)
    assert reloaded.analysis.detectors == config.analysis.detectors
    assert reloaded.detector_registry.detectors == config.detector_registry.detectors
    assert config.analysis.network is not None
    assert (
        reloaded.detector_registry.networks[config.analysis.network]
        == config.analysis.detectors
    )


def test_detector_overrides_do_not_change_catalog_keys() -> None:
    raw = example_raw()
    before = build_run_config(raw)
    after = build_run_config(
        raw,
        detectors={
            "S1": {"geometry": {"xarm_azimuth_rad": 0.3}, "psd_reference": "ET_D_psd"}
        },
    )
    for role in ("injection", "proposal"):
        assert before.catalog_request(role).key() == after.catalog_request(role).key()


def test_layer_precedence_and_figure_registries(tmp_path: Path) -> None:
    write_defaults(tmp_path, networks={"custom": ["E1", "E2"]})
    shared = tmp_path / "config/detectors.toml"
    shared.write_text(
        shared.read_text()
        + "\n[detectors.E1.geometry]\nxarm_azimuth_rad = 0.31\nelevation_m = 80.0\n"
    )
    directory = tmp_path / "config/runs/demo"
    directory.mkdir(parents=True)
    (directory / "_base.toml").write_text(
        '[analysis]\nnetwork = "custom"\n[detectors.E1.geometry]\nxarm_azimuth_rad = 0.3\n'
    )
    (directory / "one.toml").write_text(
        "[detectors.E1.geometry]\nxarm_azimuth_rad = 0.32\n"
    )
    (directory / "two.toml").write_text('[detectors.E1]\npsd_reference = "ET_D_psd"\n')
    first = build_run_config(assemble_run("demo", "one", root=tmp_path))
    second = build_run_config(assemble_run("demo", "two", root=tmp_path))
    assert first.detector_registry.detectors["E1"].geometry.xarm_azimuth_rad == 0.32
    assert second.detector_registry.detectors["E1"].geometry.xarm_azimuth_rad == 0.3
    assert first.detector_registry.detectors["E1"].geometry.elevation_m == 80.0
    resolved = resolve_networks(
        [("demo", "one"), ("demo", "two")],
        [("one", "One"), ("two", "Two")],
        root=tmp_path,
    )
    assert resolved[0].detector_registry == first.detector_registry
    assert resolved[1].detector_registry == second.detector_registry


def test_external_psd_inputs_follow_selection_and_resolution(tmp_path: Path) -> None:
    path = tmp_path / "psd.txt"
    path.write_text("10 1e-46\n20 2e-46\n")
    registry = detector_registry(
        detectors={
            "E1": {"psd_reference": str(path)},
            "S1": {"psd_reference": "https://example.org/psd.txt"},
        }
    )
    assert registry.local_psd_inputs(registry.networks["ET-triangular"]) == (path,)
    assert registry.local_psd_inputs(registry.networks["ET-2L-aligned"]) == ()
    assert detector_registry().local_psd_inputs(["S1", "R1"]) == ()


def test_registry_validation_and_build_leave_backend_configurable() -> None:
    code = """
import sys
from astrogwb.paper.config import detector_registry
registry = detector_registry()
assert 'jax' not in sys.modules
registry.local_psd_inputs(['E1', 'E2'])
assert 'jax' not in sys.modules
registry.build_network('ET-triangular')
import numpyro
numpyro.set_host_device_count(2)
import jax
assert jax.device_count() == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_snr_uses_each_networks_own_detector_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astrogwb.paper import catalogs, snr
    from astrogwb.paper.plotting import Network

    path = tmp_path / "psd.txt"
    path.write_text("5 1e-46\n10 1e-46\n50 1e-46\n100 1e-46\n")
    noisier = tmp_path / "noisier.txt"
    noisier.write_text("5 2e-46\n10 2e-46\n50 2e-46\n100 2e-46\n")
    members = ("S1", "R1")
    registries = [
        detector_registry(
            detectors={name: {"psd_reference": str(reference)} for name in members}
        )
        for reference in (path, noisier)
    ]
    networks = [
        Network(str(i), str(i), members, registry)
        for i, registry in enumerate(registries)
    ]
    monkeypatch.setattr(catalogs, "load_run_catalog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        snr,
        "prepare_observation",
        lambda *_args, **_kwargs: SimpleNamespace(
            frequencies=np.array([10.0, 50.0]),
            frequency_mask=np.array([True, True]),
            spectral_density=np.array([1e-46, 1e-46]),
            df=40.0,
        ),
    )
    result = snr.compute_network_snrs(
        path,
        networks,
        {},
        observation_time=1.0,
        minimum_redshift=0.0,
        maximum_redshift=1.0,
        minimum_frequency=5.0,
        maximum_frequency=100.0,
    )
    assert result.iloc[0].snr == pytest.approx(2.0 * result.iloc[1].snr)


@pytest.mark.parametrize(
    "network",
    [
        "ET-triangular",
        "ET-triangular-CE-Hanford",
        "ET-2L-aligned",
        "ET-2L-aligned-CE-Hanford",
        "ET-2L-misaligned",
        "ET-2L-misaligned-CE-Hanford",
    ],
)
def test_network_outputs_preserved_from_before_gwmock_migration(network: str) -> None:
    from astrogwb.detector import effective_psd, pairwise_overlap_reduction_function

    # Captured from the unmodified degree-table implementation on the same grid.
    path = Path(__file__).parent / "fixtures/detector_networks_before_gwmock.npz"
    geometry, sensitivities = detector_registry().build_network(network)
    with np.load(path) as reference:
        frequencies = reference["frequencies"]
        actual = {
            "orf": pairwise_overlap_reduction_function(frequencies, geometry),
            "psd": effective_psd(frequencies, geometry, sensitivities),
        }
        for kind, result in actual.items():
            expected = reference[f"{network}_{kind}"]
            np.testing.assert_array_equal(np.isfinite(result), np.isfinite(expected))
            np.testing.assert_allclose(
                result, expected, rtol=1e-10, atol=1e-10 if kind == "orf" else 0.0
            )


def test_legacy_saved_registry_translates_then_saves_canonical_fields(
    tmp_path: Path,
) -> None:
    config = build_run_config(example_raw())
    saved = config.model_dump(mode="json")
    path = Path(__file__).parent / "fixtures/legacy_detector_registry.json"
    legacy = json.loads(path.read_text())
    # Preserve an old custom setting instead of merging current defaults over it.
    legacy["detectors"]["S1"]["geometry"]["xarm_azimuth"] = -270.0
    saved["detector_registry"] = legacy
    before = json.dumps(saved)
    restored = build_run_config(saved)
    assert json.dumps(saved) == before
    geometry = restored.detector_registry.detectors["S1"].geometry
    assert geometry.xarm_azimuth_rad == 0.0
    for name, detector in config.detector_registry.detectors.items():
        if name != "S1":
            assert restored.detector_registry.detectors[name] == detector
    for role in ("injection", "proposal"):
        assert (
            restored.catalog_request(role).key() == config.catalog_request(role).key()
        )
    output = tmp_path / "canonical.json"
    restored.save(output)
    canonical = json.loads(output.read_text())
    assert set(canonical["detector_registry"]["detectors"]["S1"]["geometry"]) == {
        "latitude_rad",
        "longitude_rad",
        "elevation_m",
        "xarm_azimuth_rad",
        "yarm_azimuth_rad",
        "xarm_tilt_rad",
        "yarm_tilt_rad",
    }
    assert build_run_config(canonical).model_dump(mode="json") == restored.model_dump(
        mode="json"
    )


@pytest.mark.parametrize(
    "geometry",
    [{"xarm_azimuth": 72.0}, {"xarm_azimuth": 72.0, "yarm_azimuth_rad": 0.4}],
)
def test_new_toml_overrides_reject_legacy_or_mixed_keys(
    tmp_path: Path, geometry: dict
) -> None:
    from astrogwb.paper.config.runs import merge_config_layers

    path = tmp_path / "detectors.toml"
    fields = "\n".join(f"{key} = {value}" for key, value in geometry.items())
    path.write_text(f"[detectors.E1.geometry]\n{fields}\n")
    merged = merge_config_layers([path])
    from astrogwb.paper.config.detectors import DetectorRegistry

    with pytest.raises(ValidationError, match="legacy geometry fields.*_rad/_m"):
        DetectorRegistry.model_validate(merged)


def test_saved_registry_rejects_mixed_geometry_fields() -> None:
    saved = build_run_config(example_raw()).model_dump(mode="json")
    saved["detector_registry"]["detectors"]["E1"]["geometry"]["latitude"] = 43.63
    with pytest.raises(ValidationError, match="mixed legacy and gwmock"):
        build_run_config(saved)


@pytest.mark.parametrize(
    "geometry",
    [
        {"latitude_rad": 2.0},
        {"longitude_rad": 4.0},
        {"elevation_m": 1e6},
        {"xarm_azimuth_rad": float("inf")},
        {"xarm_tilt_rad": float("nan")},
    ],
)
def test_geometry_validation_matches_native_bounds(geometry: dict) -> None:
    with pytest.raises(ValidationError):
        detector_registry(detectors={"E1": {"geometry": geometry}})
