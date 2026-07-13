from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_mcmc_configs import NetworkSpec, SweepSpec, generate_configs

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "mcmc.example.toml"


def _sweep_spec() -> SweepSpec:
    return SweepSpec(
        networks={
            "net-a": NetworkSpec(detectors=("S1", "R1"), f_min=10.0, f_max=2000.0),
            "net-b": NetworkSpec(detectors=("S1", "R1", "C1")),
        },
        priors={
            "H0": {"type": "uniform", "low": 20.0, "high": 140.0},
            "Omega_m": {"type": "uniform", "low": 0.05, "high": 0.95},
        },
        campaigns={
            "cosmology": {
                "H0": (("H0",), {}),
                "H0-Omega_m": (("H0", "Omega_m"), {}),
            },
        },
    )


def test_generate_configs_applies_per_network_band_override(tmp_path: Path) -> None:
    output_dir, written, _skipped, _written_manifests, _skipped_manifests = (
        generate_configs(
            tmp_path / "mcmc",
            example_config=EXAMPLE_CONFIG,
            sweep_spec=_sweep_spec(),
            manifest_dir=tmp_path,
        )
    )
    assert len(written) == 4

    overridden = json.loads(
        (output_dir / "cosmology" / "net-a__H0.json").read_text(encoding="utf-8")
    )
    assert overridden["analysis"]["f_min"] == 10.0
    assert overridden["analysis"]["f_max"] == 2000.0

    inherited = json.loads(
        (output_dir / "cosmology" / "net-b__H0.json").read_text(encoding="utf-8")
    )
    assert inherited["analysis"]["f_min"] == 2.0
    assert inherited["analysis"]["f_max"] == 4096.0


def test_generate_configs_skips_manifests_by_default(tmp_path: Path) -> None:
    result = generate_configs(
        tmp_path / "mcmc",
        example_config=EXAMPLE_CONFIG,
        sweep_spec=_sweep_spec(),
        manifest_dir=tmp_path,
    )
    _output_dir, written, _skipped, written_manifests, skipped_manifests = result

    assert len(written) == 4
    assert written_manifests == []
    assert skipped_manifests == []
    assert not list(tmp_path.glob("mcmc.batch.*.json"))


def test_write_manifests_lists_every_campaign_run_in_order(tmp_path: Path) -> None:
    result = generate_configs(
        tmp_path / "mcmc",
        example_config=EXAMPLE_CONFIG,
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
    # jax_platforms is a runtime concern owned by the Snakemake profile, not
    # the MCMC campaign -- it must never appear in a generated manifest.
    assert "jax_platforms" not in manifest
    assert [run["campaign"] for run in manifest["runs"]] == ["cosmology"] * 4

    # Network-outer, sample-label-inner order, matching the JSON generation loop.
    assert [Path(run["config"]).name for run in manifest["runs"]] == [
        "net-a__H0.json",
        "net-a__H0-Omega_m.json",
        "net-b__H0.json",
        "net-b__H0-Omega_m.json",
    ]
    assert {Path(run["config"]).name for run in manifest["runs"]} == {
        p.name for p in written
    }


def test_manifest_respects_skip_existing_and_force(tmp_path: Path) -> None:
    manifest_path = tmp_path / "mcmc.batch.cosmology.json"

    generate_configs(
        tmp_path / "mcmc",
        example_config=EXAMPLE_CONFIG,
        sweep_spec=_sweep_spec(),
        write_manifests=True,
        manifest_dir=tmp_path,
    )
    original_text = manifest_path.read_text(encoding="utf-8")

    # Mutate the manifest to prove skip_existing leaves it untouched.
    manifest_path.write_text("mutated: true\n", encoding="utf-8")
    result = generate_configs(
        tmp_path / "mcmc",
        example_config=EXAMPLE_CONFIG,
        sweep_spec=_sweep_spec(),
        write_manifests=True,
        manifest_dir=tmp_path,
    )
    _output_dir, _written, _skipped, written_manifests, skipped_manifests = result
    assert written_manifests == []
    assert skipped_manifests == [manifest_path]
    assert manifest_path.read_text(encoding="utf-8") == "mutated: true\n"

    # skip_existing=False (== --force) overwrites it back to the generated content.
    generate_configs(
        tmp_path / "mcmc",
        example_config=EXAMPLE_CONFIG,
        sweep_spec=_sweep_spec(),
        skip_existing=False,
        write_manifests=True,
        manifest_dir=tmp_path,
    )
    assert manifest_path.read_text(encoding="utf-8") == original_text
