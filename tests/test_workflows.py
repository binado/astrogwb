from __future__ import annotations

import json
from pathlib import Path
import subprocess


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
        catalog,
        [
            {"campaign": "first-campaign", "config": str(run_config)},
            {"campaign": "second-campaign", "config": str(second_run_config)},
        ],
    )

    result = _snakemake(
        "--snakefile",
        "workflow/mcmc.smk",
        "--configfile",
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
        catalog,
        [
            {"campaign": "test-campaign", "config": str(first_config)},
            {"campaign": "test-campaign", "config": str(second_config)},
        ],
    )

    result = _snakemake(
        "--snakefile",
        "workflow/mcmc.smk",
        "--configfile",
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
        tmp_path / "missing.h5",
        [{"campaign": "test-campaign", "config": str(run_config)}],
    )

    result = _snakemake(
        "--snakefile",
        "workflow/mcmc.smk",
        "--configfile",
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
        "mcmc_compare_posteriors",
        "paper_figures",
        "snr_by_detector",
    }


def _write_manifest(tmp_path: Path, catalog: Path, runs: list[dict[str, str]]) -> Path:
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "catalog": {"id": "test-catalog", "path": str(catalog)},
                "chains_dir": str(tmp_path / "chains"),
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    return manifest
