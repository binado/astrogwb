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


def test_plot_cosmological_parameters_expands_all_chains_and_figures(
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
        "plot_cosmological_parameters",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule assemble_config:") == 1
    assert result.stdout.count("rule run_mcmc:") == 9
    assert result.stdout.count("rule plot_cosmological_parameters:") == 1
    for path in (
        "outputs/figures/cosmological-parameters/H0-by-detector.pdf",
        "outputs/figures/cosmological-parameters/H0-merger-rate-priors.pdf",
        "outputs/figures/cosmological-parameters/H0-merger-rate-corner.pdf",
        "outputs/figures/cosmological-parameters/H0-Omega_m-corner.pdf",
        "outputs/figures/cosmological-parameters/H0-Omega_m-ess-corner.pdf",
    ):
        assert path in result.stdout


def test_chains_only_target_excludes_figure_rule(tmp_path: Path) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-df1.h5")

    result = _snakemake(
        "--snakefile",
        str(MCMC_SNAKEFILE),
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "cosmological_parameters_chains",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 9
    assert "rule plot_cosmological_parameters:" not in result.stdout


def test_config_assembly_reads_the_single_inventory(
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
        "outputs/chains/cosmological-parameters/H0-Omega_m.nc",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "astrogwb-validate-config inputs/config.yaml "
        "--output-dir outputs/configs" in result.stdout
    )
    assert result.stdout.count("rule assemble_config:") == 1


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
        "cosmological_parameters_chains",
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
        "cosmological_parameters_chains",
        "plot_cosmological_parameters",
        "modified_propagation",
        "astrophysical_parameters",
        "variable_injection_size",
        "amplitude_toy",
        "fiducial_spectrum",
        "importance_weights_grid",
        "assemble_config",
        "run_mcmc",
    } <= rules
    assert {
        "H0_all_detectors_chains",
        "H0_merger_rate_chains",
        "H0_omega_m_chains",
        "modified_propagation_all_detectors",
        "star_formation_peak",
        "H0_all_detectors",
        "H0_merger_rate",
        "H0_omega_m",
        "plot_H0_all_detectors",
        "plot_H0_merger_rate",
        "plot_H0_omega_m",
        "standalone_figures",
    }.isdisjoint(rules)


def test_experiments_target_builds_all_22_chains(tmp_path: Path) -> None:
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
        "experiments",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule assemble_config:") == 1
    assert result.stdout.count("rule run_mcmc:") == 22
    assert result.stdout.count("rule plot_cosmological_parameters:") == 1
    assert result.stdout.count("rule plot_modified_propagation:") == 1


def test_plot_cosmological_parameters_passes_all_paths_not_labels(
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
        "plot_cosmological_parameters",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert "rule plot_cosmological_parameters:" in result.stdout
    assert "--base-config inputs/config.yaml" in result.stdout
    for flag in (
        "--catalog",
        "--detector-chains",
        "--prior-chains",
        "--omega-m-chain",
        "--output-detector-pdf",
        "--output-detector-csv",
        "--output-detector-tex",
        "--output-prior-pdf",
        "--output-narrow-corner-pdf",
        "--output-merger-rate-csv",
        "--output-merger-rate-tex",
        "--output-omega-m-corner-pdf",
        "--output-omega-m-ess-corner-pdf",
    ):
        assert flag in result.stdout
    assert "--section" not in result.stdout
    # The script hard-codes its own labels and run order, so neither a
    # figure config nor any LaTeX crosses the shell boundary.
    assert "--figure-config" not in result.stdout
    assert "--prior-labels" not in result.stdout
    assert r"\mathcal" not in result.stdout


def test_standalone_figures_receive_config_paths(
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
        "amplitude_toy",
        "fiducial_spectrum",
        "importance_weights_grid",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    for script in (
        "scripts/amplitude_toy_model.py",
        "scripts/fiducial_spectrum.py",
        "scripts/importance_weights_grid.py",
    ):
        assert script in result.stdout
    # Every standalone script reads the base config for itself instead of
    # receiving fiducials and analysis bounds as reconstructed flags.
    assert result.stdout.count("--base-config inputs/config.yaml") == 3
    assert "--figure-config" not in result.stdout
    for flag in ("--observation-time", "--f-min", "--h0", "--omega-gw-min"):
        assert flag not in result.stdout


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
        "outputs/figures/cosmological-parameters/H0-by-detector.pdf",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert "rule plot_cosmological_parameters:" in result.stdout
    assert result.stdout.count("rule run_mcmc:") == 9


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
        "plot_cosmological_parameters",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    command = result.stdout[result.stdout.index(" --detector-chains ") :]
    positions = [
        command.index(f"outputs/chains/cosmological-parameters/{run}.nc")
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
    assert command.index(
        "outputs/chains/cosmological-parameters/fixed.nc"
    ) < command.index("outputs/chains/cosmological-parameters/sampled.nc")
