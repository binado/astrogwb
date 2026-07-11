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


def test_mcmc_workflow_uses_only_manifest_inputs(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.h5"
    run_config = tmp_path / "selected-run.json"
    catalog.touch()
    run_config.write_text("{}\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path, catalog, run_config)

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
    assert "rule run_mcmc:" in result.stdout
    assert str(catalog) in result.stdout
    assert str(run_config) in result.stdout
    assert "rule bns_population:" not in result.stdout
    assert "rule bns_waveform_catalog:" not in result.stdout


def test_mcmc_workflow_does_not_generate_missing_catalog(tmp_path: Path) -> None:
    run_config = tmp_path / "selected-run.json"
    run_config.write_text("{}\n", encoding="utf-8")
    manifest = _write_manifest(tmp_path, tmp_path / "missing.h5", run_config)

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


def _write_manifest(tmp_path: Path, catalog: Path, run_config: Path) -> Path:
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "catalog": {"id": "test-catalog", "path": str(catalog)},
                "chains_dir": str(tmp_path / "chains"),
                "runs": [{"campaign": "test-campaign", "config": str(run_config)}],
            }
        ),
        encoding="utf-8",
    )
    return manifest
