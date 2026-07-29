from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _snakemake(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["snakemake", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_catalog_workflow_dry_run_contains_both_local_stages() -> None:
    result = _snakemake(
        "--snakefile",
        "workflow/catalog.smk",
        "--dry-run",
        "--forceall",
        "--cores",
        "1",
        "out/catalogs/bns-n8192-df1.h5",
    )

    assert result.returncode == 0, result.stderr
    assert "rule bns_population:" in result.stdout
    assert "rule bns_waveform_catalog:" in result.stdout
    assert "rule run_mcmc:" not in result.stdout
    assert "configs/catalogs/bns-n16384-df1.toml" not in result.stdout


def test_mcmc_workflow_expands_multiple_manifest_runs(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.h5"
    run_config = tmp_path / "selected-run.json"
    second_run_config = tmp_path / "second-run.toml"
    catalog.touch()
    run_config.write_text("{}\n", encoding="utf-8")
    second_run_config.write_text("", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [
            {"campaign": "first-campaign", "config": str(run_config)},
            {"campaign": "second-campaign", "config": str(second_run_config)},
        ],
    )
    catalog_config = _write_catalog_config(tmp_path, catalog)

    result = _snakemake(
        "--snakefile",
        "workflow/mcmc.smk",
        "--configfile",
        str(catalog_config),
        str(manifest),
        "--dry-run",
        "--forceall",
        "--cores",
        "1",
        "mcmc",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 2
    assert str(catalog) in result.stdout
    assert str(run_config) in result.stdout
    assert str(second_run_config) in result.stdout
    assert (
        str(tmp_path / "chains/test-catalog/first-campaign/selected-run.nc")
        in result.stdout
    )
    assert (
        str(tmp_path / "chains/test-catalog/second-campaign/second-run.nc")
        in result.stdout
    )
    assert "rule bns_population:" not in result.stdout
    assert "rule bns_waveform_catalog:" not in result.stdout


def test_mcmc_catalog_id_override_uses_derived_path(tmp_path: Path) -> None:
    catalog_id = f"test-{tmp_path.name}"
    catalog = REPO_ROOT / "out" / "catalogs" / f"{catalog_id}.h5"
    run_config = tmp_path / "selected-run.json"
    run_config.write_text("{}\n", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [{"campaign": "test-campaign", "config": str(run_config)}],
    )
    catalog_config = tmp_path / "catalog-config.json"
    catalog_config.write_text(
        json.dumps({"catalog": {"id": catalog_id}}),
        encoding="utf-8",
    )

    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.touch()
    try:
        result = _snakemake(
            "--snakefile",
            "workflow/mcmc.smk",
            "--configfile",
            str(catalog_config),
            str(manifest),
            "--dry-run",
            "--forceall",
            "--cores",
            "1",
            "mcmc",
        )
    finally:
        catalog.unlink()

    assert result.returncode == 0, result.stderr
    assert f"out/catalogs/{catalog_id}.h5" in result.stdout
    assert "out/catalogs/bns-n16384-df1.h5" not in result.stdout
    assert str(tmp_path / f"chains/{catalog_id}/test-campaign/selected-run.nc") in (
        result.stdout
    )


def test_mcmc_thread_override_controls_runner_cpu_budget(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.h5"
    run_config = tmp_path / "selected-run.json"
    catalog.touch()
    run_config.write_text("{}\n", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [{"campaign": "test-campaign", "config": str(run_config)}],
    )
    catalog_config = _write_catalog_config(tmp_path, catalog)

    result = _snakemake(
        "--snakefile",
        "workflow/mcmc.smk",
        "--configfile",
        str(catalog_config),
        str(manifest),
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "2",
        "mcmc",
        "--set-threads",
        "run_mcmc=2",
    )

    assert result.returncode == 0, result.stderr
    assert "threads: 2" in result.stdout
    assert "--cpu-threads 2" in result.stdout
    assert "--platform cuda" in result.stdout
    assert "cpus_per_task" not in result.stdout
    assert "OMP_NUM_THREADS" not in result.stdout
    assert "OPENBLAS_NUM_THREADS" not in result.stdout
    assert "MKL_NUM_THREADS" not in result.stdout
    assert "XLA_FLAGS" not in result.stdout


def test_mcmc_workflow_rejects_duplicate_campaign_and_config_stem(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.h5"
    first_config = tmp_path / "first" / "selected-run.json"
    second_config = tmp_path / "second" / "selected-run.toml"
    catalog.touch()
    first_config.parent.mkdir()
    second_config.parent.mkdir()
    first_config.write_text("{}\n", encoding="utf-8")
    second_config.write_text("", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [
            {"campaign": "test-campaign", "config": str(first_config)},
            {"campaign": "test-campaign", "config": str(second_config)},
        ],
    )
    catalog_config = _write_catalog_config(tmp_path, catalog)

    result = _snakemake(
        "--snakefile",
        "workflow/mcmc.smk",
        "--configfile",
        str(catalog_config),
        str(manifest),
        "--dry-run",
        "--cores",
        "1",
        "mcmc",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "duplicate MCMC run test-campaign/selected-run" in output


def test_mcmc_workflow_does_not_generate_missing_catalog(tmp_path: Path) -> None:
    run_config = tmp_path / "selected-run.json"
    run_config.write_text("{}\n", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [{"campaign": "test-campaign", "config": str(run_config)}],
    )
    catalog_config = _write_catalog_config(tmp_path, tmp_path / "missing.h5")

    result = _snakemake(
        "--snakefile",
        "workflow/mcmc.smk",
        "--configfile",
        str(catalog_config),
        str(manifest),
        "--dry-run",
        "--cores",
        "1",
        "mcmc",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "MissingInputException" in output
    assert "bns_population" not in output
    assert "bns_waveform_catalog" not in output


def test_paper_workflow_exposes_only_figure_rules() -> None:
    result = _snakemake(
        "--snakefile",
        "workflow/paper.smk",
        "--list-rules",
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.split()) == {
        "amplitude_toy",
        "fiducial_spectrum",
        "mcmc_cosmological_parameters",
        "mcmc_modified_propagation",
        "paper_figures",
    }


def _write_manifest(tmp_path: Path, runs: list[dict[str, str]]) -> Path:
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "chains_dir": str(tmp_path / "chains"),
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _write_catalog_config(tmp_path: Path, catalog: Path) -> Path:
    catalog_config = tmp_path / "catalog-config.json"
    catalog_config.write_text(
        json.dumps({"catalog": {"id": "test-catalog", "path": str(catalog)}}),
        encoding="utf-8",
    )
    return catalog_config
