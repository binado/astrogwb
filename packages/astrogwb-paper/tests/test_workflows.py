from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()
SNAKEFILE = PAPER_ROOT / "Snakefile"
CATALOG_RULES = (
    "population_config",
    "population",
    "waveform_catalog",
    "catalog_snr",
    "catalogs",
)
MCMC_RULES = (
    "assemble_config",
    "run_mcmc",
    "plot_cosmological_parameters",
    "plot_modified_propagation",
    "amplitude_toy",
    "fiducial_spectrum",
    "importance_weights_grid",
    "experiments",
    "run_experiment_cosmological_parameters",
    "run_experiment_modified_propagation",
    "run_experiment_astrophysical_parameters",
    "run_experiment_variable_catalog_size",
    "run_experiment_variable_proposal_guard",
    "run_experiment_waveform_approximant",
)


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


def _mcmc(*args: str) -> subprocess.CompletedProcess[str]:
    return _snakemake(
        "--snakefile",
        str(SNAKEFILE),
        "--allowed-rules",
        *MCMC_RULES,
        *args,
    )


def _rule_inputs(stdout: str) -> list[str]:
    """Return the ``input:`` line of every job in a dry-run report."""
    return [
        line.strip()
        for line in stdout.splitlines()
        if line.strip().startswith("input:")
    ]


def _catalogs(tmp_path: Path, *names: str) -> Path:
    directory = tmp_path / "catalogs"
    directory.mkdir()
    (directory / "injection-bns-n32768-eps=0-df1.h5").touch()
    for name in names:
        (directory / name).touch()
    return directory


def test_catalog_workflow_builds_fresh_populations_and_waveforms() -> None:
    result = _snakemake(
        "--snakefile",
        str(SNAKEFILE),
        "--allowed-rules",
        *CATALOG_RULES,
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "1",
        "outputs/catalogs/injection-bns-n32768-eps=0-df1.h5",
        "outputs/catalogs/bns-n16384-eps=0.1-df1.h5",
    )

    assert result.returncode == 0, result.stderr
    assert "inputs/catalogs.yaml" in result.stdout
    assert "outputs/population-configs/md.yaml" in result.stdout
    assert "outputs/population-configs/uniform-redshift.yaml" in result.stdout
    assert "outputs/populations/injection-bns-n32768-eps=0-df1.h5" in result.stdout
    assert "outputs/populations/bns-n16384-eps=0.1-df1.h5" in result.stdout
    assert "outputs/waveforms/injection-bns-n32768-eps=0-df1.h5" in result.stdout
    assert "outputs/waveforms/bns-n16384-eps=0.1-df1.h5" in result.stdout
    assert "outputs/catalogs/injection-bns-n32768-eps=0-df1.h5" in result.stdout
    assert "outputs/catalogs/bns-n16384-eps=0.1-df1.h5" in result.stdout
    assert "--uniform-mixing-fraction 0.1" in result.stdout
    # SNR enrichment covers the union of every network's detectors, so each
    # run finds its columns whatever network it aliases.
    assert "astrogwb-compute-snr" in result.stdout
    assert "--detectors E1,E2,E3,C1,S1,R1,S2,R2" in result.stdout
    assert "--batch-size 512" in result.stdout


def test_catalogs_target_builds_all_8_catalogs() -> None:
    result = _snakemake(
        "--snakefile",
        str(SNAKEFILE),
        "--allowed-rules",
        *CATALOG_RULES,
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "catalogs",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule population_config:") == 2
    assert result.stdout.count("rule waveform_catalog:") == 8
    assert result.stdout.count("rule catalog_snr:") == 8
    for name in (
        "injection-bns-n32768-eps=0-df1",
        "bns-n32768-eps=0-df1-taylorf2",
        "bns-n8192-eps=0-df1",
        "bns-n16384-eps=0-df1",
        "bns-n32768-eps=0-df1",
        "bns-n16384-eps=0.1-df1",
        "bns-n16384-eps=0.01-df1",
        "bns-n16384-eps=0.001-df1",
    ):
        assert f"outputs/catalogs/{name}.h5" in result.stdout


def test_plot_cosmological_parameters_expands_all_chains_and_figures(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-eps=0-df1.h5")

    result = _mcmc(
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
    assert result.stdout.count("rule run_mcmc:") == 8
    assert result.stdout.count("rule plot_cosmological_parameters:") == 1
    for path in (
        "outputs/figures/cosmological-parameters/H0-by-detector.pdf",
        "outputs/figures/cosmological-parameters/H0-merger-rate-priors.pdf",
        "outputs/figures/cosmological-parameters/H0-merger-rate-corner.pdf",
        "outputs/figures/cosmological-parameters/H0-Omega_m-corner.pdf",
        "outputs/figures/cosmological-parameters/H0-Omega_m-ess-corner.pdf",
    ):
        assert path in result.stdout


def test_run_experiment_target_excludes_figure_rule(tmp_path: Path) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-eps=0-df1.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "run_experiment_cosmological_parameters",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 8
    assert "rule plot_cosmological_parameters:" not in result.stdout


def test_config_assembly_reads_the_single_inventory(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-eps=0-df1.h5")

    result = _mcmc(
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
        "astrogwb-validate-config inputs/experiments.yaml "
        "--output-dir outputs/configs" in result.stdout
    )
    assert "--injection-catalog" in result.stdout
    assert "--proposal-catalog" in result.stdout
    assert str(catalogs / "injection-bns-n32768-eps=0-df1.h5") in result.stdout
    assert str(catalogs / "bns-n16384-eps=0-df1.h5") in result.stdout
    assert result.stdout.count("rule assemble_config:") == 1


def test_variable_catalog_size_uses_three_catalogs(tmp_path: Path) -> None:
    catalogs = _catalogs(
        tmp_path,
        "bns-n8192-eps=0-df1.h5",
        "bns-n16384-eps=0-df1.h5",
        "bns-n32768-eps=0-df1.h5",
    )

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "run_experiment_variable_catalog_size",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 3
    for name in (
        "bns-n8192-eps=0-df1.h5",
        "bns-n16384-eps=0-df1.h5",
        "bns-n32768-eps=0-df1.h5",
    ):
        assert str(catalogs / name) in result.stdout


def test_variable_proposal_guard_uses_three_catalogs(tmp_path: Path) -> None:
    catalogs = _catalogs(
        tmp_path,
        "bns-n16384-eps=0.1-df1.h5",
        "bns-n16384-eps=0.01-df1.h5",
        "bns-n16384-eps=0.001-df1.h5",
    )

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "run_experiment_variable_proposal_guard",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 3
    for name in (
        "bns-n16384-eps=0.1-df1.h5",
        "bns-n16384-eps=0.01-df1.h5",
        "bns-n16384-eps=0.001-df1.h5",
    ):
        assert str(catalogs / name) in result.stdout


def test_waveform_approximant_uses_imr_data_and_two_proposals(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n32768-eps=0-df1-taylorf2.h5")
    imr = catalogs / "injection-bns-n32768-eps=0-df1.h5"
    taylorf2 = catalogs / "bns-n32768-eps=0-df1-taylorf2.h5"

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "8",
        "run_experiment_waveform_approximant",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 2
    assert result.stdout.count(f"--injection-catalog {imr}") == 2
    assert result.stdout.count(f"--proposal-catalog {imr}") == 1
    assert result.stdout.count(f"--proposal-catalog {taylorf2}") == 1


def test_missing_catalog_does_not_acquire_a_producer(tmp_path: Path) -> None:
    catalogs = tmp_path / "missing-catalogs"

    result = _mcmc(
        "--dry-run",
        "--cores",
        "4",
        "run_experiment_cosmological_parameters",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "MissingInputException" in output
    assert "population_config" not in output
    assert "waveform_catalog" not in output
    assert "catalog_snr" not in output


def test_unified_workflow_exposes_explicit_experiment_targets() -> None:
    result = _snakemake("--snakefile", str(SNAKEFILE), "--list-rules")

    assert result.returncode == 0, result.stderr
    rules = set(result.stdout.split())
    assert {
        "run_experiment_cosmological_parameters",
        "run_experiment_modified_propagation",
        "run_experiment_astrophysical_parameters",
        "run_experiment_variable_catalog_size",
        "run_experiment_variable_proposal_guard",
        "run_experiment_waveform_approximant",
        "catalogs",
        "catalog_snr",
        "plot_cosmological_parameters",
        "amplitude_toy",
        "fiducial_spectrum",
        "importance_weights_grid",
        "assemble_config",
        "run_mcmc",
    } <= rules
    assert {
        # The figure outputs used to hide behind the bare experiment target;
        # with them gone the bare names must not come back either.
        "cosmological_parameters",
        "cosmological_parameters_chains",
        "modified_propagation",
        "modified_propagation_chains",
        "astrophysical_parameters",
        "astrophysical_parameters_chains",
        "variable_proposal_size",
        "variable_proposal_size_chains",
        "variable_catalog_size",
        "variable_catalog_size_chains",
        "variable_proposal_guard",
        "variable_proposal_guard_chains",
        "waveform_approximant",
        "waveform_approximant_chains",
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


def test_experiments_target_builds_all_26_chains(tmp_path: Path) -> None:
    catalogs = _catalogs(
        tmp_path,
        "bns-n8192-eps=0-df1.h5",
        "bns-n16384-eps=0-df1.h5",
        "bns-n32768-eps=0-df1.h5",
        "bns-n16384-eps=0.1-df1.h5",
        "bns-n16384-eps=0.01-df1.h5",
        "bns-n16384-eps=0.001-df1.h5",
        "bns-n32768-eps=0-df1-taylorf2.h5",
    )

    result = _mcmc(
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
    assert result.stdout.count("rule run_mcmc:") == 26
    # `experiments` is chains-only now; figures are opt-in via the plot rules.
    assert "rule plot_cosmological_parameters:" not in result.stdout
    assert "rule plot_modified_propagation:" not in result.stdout


def test_plot_cosmological_parameters_passes_all_paths_not_labels(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-eps=0-df1.h5")

    result = _mcmc(
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
    # The script resolves the inventory path itself, so no flag carries it --
    # but the rule still declares the file, so editing it retriggers the figure.
    assert "--base-config" not in result.stdout
    assert any(
        "inputs/experiments.yaml" in line for line in _rule_inputs(result.stdout)
    )
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
    catalogs = _catalogs(tmp_path, "bns-n16384-eps=0-df1.h5")

    result = _mcmc(
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
    # receiving fiducials and analysis bounds as reconstructed flags -- or even
    # the inventory path, which the library already owns.
    assert "--base-config" not in result.stdout
    assert (
        sum("inputs/experiments.yaml" in line for line in _rule_inputs(result.stdout))
        == 3
    )
    assert "--figure-config" not in result.stdout
    for flag in ("--observation-time", "--f-min", "--h0", "--omega-gw-min"):
        assert flag not in result.stdout


def test_figure_path_is_a_valid_snakemake_target(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-eps=0-df1.h5")

    result = _mcmc(
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
    assert result.stdout.count("rule run_mcmc:") == 8


def test_figure_rule_preserves_declared_chain_order(tmp_path: Path) -> None:
    catalogs = _catalogs(tmp_path, "bns-n16384-eps=0-df1.h5")

    result = _mcmc(
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
    prior_start = command.index(" --prior-chains ")
    assert command.index(
        "outputs/chains/cosmological-parameters/ET-2L-aligned-CE-Hanford.nc",
        prior_start,
    ) < command.index(
        "outputs/chains/cosmological-parameters/H0-merger-rate.nc",
        prior_start,
    )
