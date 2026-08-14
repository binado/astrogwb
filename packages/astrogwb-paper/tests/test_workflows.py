from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()


def _snakemake(*args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="astrogwb-snakemake-") as cache:
        return subprocess.run(
            ["snakemake", *args],
            cwd=PAPER_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "XDG_CACHE_HOME": cache},
        )


def test_catalog_workflow_dry_run_contains_both_local_stages() -> None:
    result = _snakemake(
        "--snakefile",
        str(PAPER_ROOT / "workflow/catalog.smk"),
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


def test_mcmc_workflow_expands_a_campaign_into_config_and_chain_rules(
    tmp_path: Path,
) -> None:
    """Config assembly and sampling are one DAG: no manifest, no generator."""
    catalog = tmp_path / "catalog.h5"
    catalog.touch()
    workflow_config = _write_workflow_config(tmp_path, catalog, ["cosmology"])

    result = _snakemake(
        "--snakefile",
        str(PAPER_ROOT / "workflow/mcmc.smk"),
        "--configfile",
        str(workflow_config),
        "--dry-run",
        "--forceall",
        "--cores",
        "1",
        "mcmc",
    )

    assert result.returncode == 0, result.stderr
    # cosmology: 2 networks x 3 analyses x 1 observation.
    assert result.stdout.count("rule run_mcmc:") == 6
    assert result.stdout.count("rule mcmc_config:") == 6
    assert str(catalog) in result.stdout
    assert (
        str(tmp_path / "chains/test-catalog/cosmology/ET-2L-aligned__H0__baseline.nc")
        in result.stdout
    )
    assert "rule bns_population:" not in result.stdout
    assert "rule bns_waveform_catalog:" not in result.stdout


def test_mcmc_config_rule_merges_fragments_through_the_validator(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.h5"
    catalog.touch()
    workflow_config = _write_workflow_config(tmp_path, catalog, ["cosmology"])

    result = _snakemake(
        "--snakefile",
        str(PAPER_ROOT / "workflow/mcmc.smk"),
        "--configfile",
        str(workflow_config),
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "1",
        "mcmc",
    )

    assert result.returncode == 0, result.stderr
    assert "knf configs/mcmc/fragments/base.toml" in result.stdout
    assert "configs/mcmc/fragments/priors.toml" in result.stdout
    assert "configs/mcmc/fragments/analyses/H0.toml" in result.stdout
    assert "--strict -f json" in result.stdout
    assert "astrogwb-validate-config -" in result.stdout


def test_mcmc_workflow_defaults_to_every_campaign(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.h5"
    catalog.touch()
    workflow_config = _write_workflow_config(tmp_path, catalog, [])

    result = _snakemake(
        "--snakefile",
        str(PAPER_ROOT / "workflow/mcmc.smk"),
        "--configfile",
        str(workflow_config),
        "--dry-run",
        "--forceall",
        "--cores",
        "1",
        "mcmc",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 58


def test_mcmc_workflow_rejects_unknown_campaign(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.h5"
    catalog.touch()
    workflow_config = _write_workflow_config(tmp_path, catalog, ["not-a-campaign"])

    result = _snakemake(
        "--snakefile",
        str(PAPER_ROOT / "workflow/mcmc.smk"),
        "--configfile",
        str(workflow_config),
        "--dry-run",
        "--cores",
        "1",
        "mcmc",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "unknown campaigns ['not-a-campaign']" in output


def test_mcmc_catalog_id_override_uses_derived_path(tmp_path: Path) -> None:
    catalog_id = f"test-{tmp_path.name}"
    catalog = PAPER_ROOT / "out" / "catalogs" / f"{catalog_id}.h5"
    workflow_config = tmp_path / "workflow-config.json"
    workflow_config.write_text(
        json.dumps(
            {
                "catalog": {"id": catalog_id},
                "chains_dir": str(tmp_path / "chains"),
                "campaigns": ["cosmology"],
            }
        ),
        encoding="utf-8",
    )

    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.touch()
    try:
        result = _snakemake(
            "--snakefile",
            str(PAPER_ROOT / "workflow/mcmc.smk"),
            "--configfile",
            str(workflow_config),
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
    assert str(tmp_path / f"chains/{catalog_id}/cosmology") in result.stdout


def test_mcmc_thread_override_controls_runner_cpu_budget(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.h5"
    catalog.touch()
    workflow_config = _write_workflow_config(tmp_path, catalog, ["cosmology"])

    result = _snakemake(
        "--snakefile",
        str(PAPER_ROOT / "workflow/mcmc.smk"),
        "--configfile",
        str(workflow_config),
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


def test_mcmc_workflow_does_not_generate_missing_catalog(tmp_path: Path) -> None:
    workflow_config = _write_workflow_config(
        tmp_path, tmp_path / "missing.h5", ["cosmology"]
    )

    result = _snakemake(
        "--snakefile",
        str(PAPER_ROOT / "workflow/mcmc.smk"),
        "--configfile",
        str(workflow_config),
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
        str(PAPER_ROOT / "workflow/paper.smk"),
        "--list-rules",
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.split()) == {
        "amplitude_toy",
        "fiducial_spectrum",
        "importance_weights_grid",
        "mcmc_cosmological_parameters",
        "mcmc_modified_propagation",
        "paper_figures",
    }


def _write_workflow_config(tmp_path: Path, catalog: Path, campaigns: list[str]) -> Path:
    workflow_config = tmp_path / "workflow-config.json"
    workflow_config.write_text(
        json.dumps(
            {
                "catalog": {"id": "test-catalog", "path": str(catalog)},
                "chains_dir": str(tmp_path / "chains"),
                "campaigns": campaigns,
            }
        ),
        encoding="utf-8",
    )
    return workflow_config
