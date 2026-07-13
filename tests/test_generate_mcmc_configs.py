from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_mcmc_configs import (
    AnalysisSpec,
    NetworkSpec,
    ObservationSpec,
    RunSpec,
    SweepConfig,
    generate_configs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "mcmc.example.toml"


def _sweep_spec(*, base_config: Path = EXAMPLE_CONFIG) -> SweepConfig:
    return SweepConfig(
        base_config=base_config,
        networks={
            "net-a": NetworkSpec(detectors=("S1", "R1")),
            "net-b": NetworkSpec(detectors=("S1", "R1", "C1")),
        },
        observations={
            "baseline": ObservationSpec(
                observation_time=1.0,
                f_min=2.0,
                f_max=4096.0,
            ),
            "narrow": ObservationSpec(
                observation_time=2.0,
                f_min=10.0,
                f_max=2000.0,
            ),
        },
        analyses={
            "H0": AnalysisSpec(
                sampled_params=("H0",),
                priors={"H0": "uniform"},
                fiducials={"H0": 70.0, "Omega_m": 0.4},
            ),
            "H0-Omega_m": AnalysisSpec(
                sampled_params=("H0", "Omega_m"),
                priors={"H0": "uniform", "Omega_m": "uniform"},
            ),
        },
        priors={
            "H0": {"uniform": {"type": "uniform", "low": 20.0, "high": 140.0}},
            "Omega_m": {"uniform": {"type": "uniform", "low": 0.05, "high": 0.95}},
        },
        runs={
            "cosmology": RunSpec(
                networks=("net-a", "net-b"),
                analyses=("H0", "H0-Omega_m"),
                observations=("baseline", "narrow"),
            )
        },
    )


def test_generate_configs_materializes_cartesian_product(tmp_path: Path) -> None:
    output_dir, written, _skipped, _written_manifests, _skipped_manifests = (
        generate_configs(
            tmp_path / "mcmc",
            sweep_spec=_sweep_spec(),
            manifest_dir=tmp_path,
        )
    )
    assert len(written) == 8

    overridden = json.loads(
        (output_dir / "cosmology" / "net-a__H0__narrow.json").read_text(
            encoding="utf-8"
        )
    )
    assert overridden["analysis"] == {
        "detectors": ["S1", "R1"],
        "f_min": 10.0,
        "f_max": 2000.0,
    }
    assert overridden["observation_time"] == 2.0
    assert overridden["sampled_params"] == ["H0"]
    assert overridden["priors"] == {
        "H0": {"type": "uniform", "low": 20.0, "high": 140.0}
    }
    assert overridden["fiducials"]["H0"] == 70.0
    assert overridden["fiducials"]["Omega_m"] == 0.4
    assert overridden["fiducials"]["xi_0"] == 1.0

    inherited = json.loads(
        (output_dir / "cosmology" / "net-b__H0-Omega_m__baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert inherited["analysis"]["detectors"] == ["S1", "R1", "C1"]
    assert inherited["analysis"]["f_min"] == 2.0
    assert inherited["analysis"]["f_max"] == 4096.0
    assert inherited["observation_time"] == 1.0
    assert inherited["fiducials"]["H0"] == 67.66
    assert inherited["fiducials"]["Omega_m"] == 0.3096


def test_generate_configs_validates_every_point_before_writing(tmp_path: Path) -> None:
    sweep = _sweep_spec()
    invalid = sweep.model_copy(
        update={
            "analyses": {
                **sweep.analyses,
                "missing-fiducial": AnalysisSpec(
                    sampled_params=("new_parameter",),
                    priors={"new_parameter": "uniform"},
                ),
            },
            "priors": {
                **sweep.priors,
                "new_parameter": {
                    "uniform": {"type": "uniform", "low": 0.0, "high": 1.0}
                },
            },
            "runs": {
                "cosmology": RunSpec(
                    networks=("net-a",),
                    analyses=("H0", "missing-fiducial"),
                    observations=("baseline",),
                )
            },
        }
    )
    output_dir = tmp_path / "mcmc"

    with pytest.raises(ValueError, match="missing from.*fiducials"):
        generate_configs(output_dir, sweep_spec=invalid, manifest_dir=tmp_path)

    assert not output_dir.exists()
    assert not list(tmp_path.glob("mcmc.batch.*.json"))


def test_generate_configs_rejects_duplicate_generated_paths(tmp_path: Path) -> None:
    sweep = _sweep_spec()
    h0 = sweep.analyses["H0"]
    ambiguous = sweep.model_copy(
        update={
            "networks": {
                "a__b": NetworkSpec(detectors=("S1", "R1")),
                "a": NetworkSpec(detectors=("S1", "R1", "C1")),
            },
            "analyses": {"c": h0, "b__c": h0},
            "runs": {
                "cosmology": RunSpec(
                    networks=("a__b", "a"),
                    analyses=("c", "b__c"),
                    observations=("baseline",),
                )
            },
        }
    )
    output_dir = tmp_path / "mcmc"

    with pytest.raises(ValueError, match="duplicate generated config path"):
        generate_configs(output_dir, sweep_spec=ambiguous, manifest_dir=tmp_path)

    assert not output_dir.exists()


def test_generate_configs_skips_manifests_by_default(tmp_path: Path) -> None:
    result = generate_configs(
        tmp_path / "mcmc",
        sweep_spec=_sweep_spec(),
        manifest_dir=tmp_path,
    )
    _output_dir, written, _skipped, written_manifests, skipped_manifests = result

    assert len(written) == 8
    assert written_manifests == []
    assert skipped_manifests == []
    assert not list(tmp_path.glob("mcmc.batch.*.json"))


def test_write_manifests_lists_every_campaign_run_in_order(tmp_path: Path) -> None:
    result = generate_configs(
        tmp_path / "mcmc",
        sweep_spec=_sweep_spec(),
        write_manifests=True,
        manifest_dir=tmp_path,
        chains_dir="test-chains",
    )
    _output_dir, written, _skipped, written_manifests, _skipped_manifests = result

    manifest_path = tmp_path / "mcmc.batch.cosmology.json"
    assert written_manifests == [manifest_path]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "catalog" not in manifest
    assert manifest["chains_dir"] == "test-chains"
    assert "jax_platforms" not in manifest
    assert [run["campaign"] for run in manifest["runs"]] == ["cosmology"] * 8

    assert [Path(run["config"]).name for run in manifest["runs"]] == [
        "net-a__H0__baseline.json",
        "net-a__H0__narrow.json",
        "net-a__H0-Omega_m__baseline.json",
        "net-a__H0-Omega_m__narrow.json",
        "net-b__H0__baseline.json",
        "net-b__H0__narrow.json",
        "net-b__H0-Omega_m__baseline.json",
        "net-b__H0-Omega_m__narrow.json",
    ]
    assert {Path(run["config"]).name for run in manifest["runs"]} == {
        path.name for path in written
    }


def test_manifest_respects_skip_existing_and_force(tmp_path: Path) -> None:
    manifest_path = tmp_path / "mcmc.batch.cosmology.json"

    generate_configs(
        tmp_path / "mcmc",
        sweep_spec=_sweep_spec(),
        write_manifests=True,
        manifest_dir=tmp_path,
    )
    original_text = manifest_path.read_text(encoding="utf-8")

    manifest_path.write_text("mutated: true\n", encoding="utf-8")
    result = generate_configs(
        tmp_path / "mcmc",
        sweep_spec=_sweep_spec(),
        write_manifests=True,
        manifest_dir=tmp_path,
    )
    _output_dir, _written, _skipped, written_manifests, skipped_manifests = result
    assert written_manifests == []
    assert skipped_manifests == [manifest_path]
    assert manifest_path.read_text(encoding="utf-8") == "mutated: true\n"

    generate_configs(
        tmp_path / "mcmc",
        sweep_spec=_sweep_spec(),
        skip_existing=False,
        write_manifests=True,
        manifest_dir=tmp_path,
    )
    assert manifest_path.read_text(encoding="utf-8") == original_text
