from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()
MCMC_SNAKEFILE = PAPER_ROOT / "workflow/mcmc.smk"
CATALOG_SNAKEFILE = PAPER_ROOT / "workflow/catalog.smk"


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


def _catalogs(tmp_path: Path, *names: str) -> Path:
    directory = tmp_path / "catalogs"
    directory.mkdir()
    for name in names:
        (directory / name).touch()
    return directory


def test_catalog_workflow_uses_input_recipes_and_output_tree() -> None:
    result = _snakemake(
        "--snakefile",
        str(CATALOG_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--cores",
        "1",
        "outputs/catalogs/bns-n8192-df1.h5",
    )

    assert result.returncode == 0, result.stderr
    assert "inputs/catalogs/bns-n8192-df1.toml" in result.stdout
    assert "outputs/populations/bns-n8192-df1.h5" in result.stdout
    assert "outputs/catalogs/bns-n8192-df1.h5" in result.stdout


def test_complete_experiment_expands_chains_and_local_figure(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-df1.h5")

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "H0_all_detectors",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule assemble_config:") == 6
    assert result.stdout.count("rule run_mcmc:") == 6
    assert result.stdout.count("rule plot_H0_all_detectors:") == 1
    assert "outputs/figures/H0-all-detectors/H0-by-detector.pdf" in result.stdout


def test_chains_only_target_excludes_figure_rule(tmp_path: Path) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-df1.h5")

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "H0_all_detectors_chains",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 6
    assert "rule plot_H0_all_detectors:" not in result.stdout


def test_config_assembly_merges_only_base_and_explicit_run(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-df1.h5")

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "4",
        "outputs/chains/H0-omega-m/H0-Omega_m.nc",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "astrogwb-validate-config --base inputs/mcmc.base.toml "
        "--run H0-Omega_m experiments/H0-omega-m.toml" in result.stdout
    )


def test_variable_injection_size_uses_three_catalogs(tmp_path: Path) -> None:
    catalogs = _catalogs(
        tmp_path,
        "bns-n8192-df1.h5",
        "bns-n16384-df1.h5",
        "bns-n32768-df1.h5",
    )

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "variable_injection_size",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 3
    for name in (
        "bns-n8192-df1.h5",
        "bns-n16384-df1.h5",
        "bns-n32768-df1.h5",
    ):
        assert str(catalogs / name) in result.stdout


def test_missing_catalog_does_not_acquire_a_producer(tmp_path: Path) -> None:
    catalogs = tmp_path / "missing-catalogs"

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--cores",
        "4",
        "H0_omega_m_chains",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "MissingInputException" in output
    assert "bns_population" not in output
    assert "bns_waveform_catalog" not in output


def test_unified_workflow_exposes_explicit_experiment_targets() -> None:
    result = _snakemake("--snakefile", str(MCMC_SNAKEFILE), "--list-rules")

    assert result.returncode == 0, result.stderr
    rules = set(result.stdout.split())
    assert {
        "H0_all_detectors",
        "H0_all_detectors_chains",
        "modified_propagation_all_detectors",
        "H0_merger_rate",
        "H0_omega_m",
        "astrophysical_parameters",
        "star_formation_peak",
        "variable_injection_size",
        "standalone_figures",
        "assemble_config",
        "run_mcmc",
    } <= rules


def test_merger_rate_figure_does_not_interpolate_latex_labels(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-df1.h5")

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "8",
        "H0_merger_rate",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert "rule plot_H0_merger_rate:" in result.stdout
    assert "--prior-labels" not in result.stdout


def test_standalone_figures_expand_parameterized_shell_commands(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-df1.h5")

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "4",
        "standalone_figures",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    for script in (
        "amplitude_toy_model.py",
        "fiducial_spectrum.py",
        "importance_weights_grid.py",
    ):
        assert script in result.stdout
    # Shared base and analysis values are injected at shell-expansion time.
    assert "--observation-time 1.0" in result.stdout
    assert "--f-min 2.0" in result.stdout
    assert "--h0 67.66" in result.stdout
    assert "--omega-gw-min 1e-15" in result.stdout


def test_figure_path_is_a_valid_snakemake_target(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-df1.h5")

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "outputs/figures/H0-all-detectors/H0-by-detector.pdf",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert "rule plot_H0_all_detectors:" in result.stdout
    assert result.stdout.count("rule run_mcmc:") == 6


def test_figure_rule_preserves_declared_chain_order(tmp_path: Path) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-df1.h5")

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "8",
        "H0_all_detectors",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    command = result.stdout[result.stdout.index(" --detector-chains ") :]
    positions = [
        command.index(f"outputs/chains/H0-all-detectors/{run}.nc")
        for run in (
            "ET-triangular",
            "ET-triangular-CE-Hanford",
            "ET-2L-aligned",
            "ET-2L-aligned-CE-Hanford",
            "ET-2L-misaligned",
            "ET-2L-misaligned-CE-Hanford",
        )
    ]
    assert positions == sorted(positions)
