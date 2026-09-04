"""Workflow-shape tests: what the DAG builds, and what each rule is handed.

Every dry run happens in a scratch directory holding symlinks to the committed
inputs (``Snakefile``, ``config/``, ``scripts/``) and nothing else. Running them
against the real checkout instead would make them depend on whichever chains
and banks a developer happens to have built -- and existing chains are
``protected()``, so a ``--forceall`` dry run against them fails outright.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from astrogwb_paper.config.runs import run_config_paths
from astrogwb_paper.paths import paper_project_root
from astrogwb_paper.plotting import DETECTOR_NETWORK_RUNS

PAPER_ROOT = paper_project_root()
SNAKEFILE = PAPER_ROOT / "Snakefile"
#: The run whose layer files the chain-free figure rules declare and pass.
FIGURE_RUN = ("cosmological-parameters", "ET-2L-aligned-CE-Hanford")
#: Those layer files as the workflow sees them, relative to its cwd. Resolved
#: against the checkout first, because `run_config_paths` checks they exist.
FIGURE_CONFIG_LAYERS = [
    str(path.relative_to(PAPER_ROOT))
    for path in run_config_paths(*FIGURE_RUN, root=PAPER_ROOT)
]
#: Set by the session fixture below; the cwd every snakemake run uses.
WORKFLOW_DIR = PAPER_ROOT

#: Committed inputs the workflow reads. Everything else it touches is output.
LINKED = ("Snakefile", "config", "scripts")
BANK_RULES = ("waveform_bank", "banks")
MCMC_RULES = (
    "validate",
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


@pytest.fixture(scope="session", autouse=True)
def _isolated_workflow_dir(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point every dry run at a scratch tree with no outputs in it."""
    global WORKFLOW_DIR
    WORKFLOW_DIR = tmp_path_factory.mktemp("workflow")
    for name in LINKED:
        (WORKFLOW_DIR / name).symlink_to(PAPER_ROOT / name)


def _snakemake(*args: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="astrogwb-snakemake-") as cache:
        return subprocess.run(
            ["snakemake", *args],
            cwd=WORKFLOW_DIR,
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


def _rule_names(stdout: str) -> set[str]:
    """Parse ``--list-rules`` output.

    Each rule is printed as ``name`` or ``name (docstring...)``, with docstring
    continuation lines indented -- so a rule name is the first token of any
    unindented, non-empty line.
    """
    return {
        line.split()[0]
        for line in stdout.splitlines()
        if line.strip() and not line[0].isspace()
    }


def _rule_inputs(stdout: str) -> list[str]:
    """Return the ``input:`` line of every job in a dry-run report."""
    return [
        line.strip()
        for line in stdout.splitlines()
        if line.strip().startswith("input:")
    ]


def _banks(tmp_path: Path, *names: str) -> Path:
    """A fake banks directory: the fixed injection bank plus any ``names``."""
    directory = tmp_path / "banks"
    directory.mkdir()
    (directory / "md-imrphenom-s41.h5").touch()
    for name in names:
        (directory / name).touch()
    return directory


def test_bank_rule_reads_its_config_and_population_directly() -> None:
    """One rule per bank; the population is no longer a workflow node."""
    result = _snakemake(
        "--snakefile",
        str(SNAKEFILE),
        "--allowed-rules",
        *BANK_RULES,
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "1",
        "outputs/banks/md-imrphenom-s41.h5",
        "outputs/banks/uniform-imrphenom-s51.h5",
    )

    assert result.returncode == 0, result.stderr
    assert "config/banks/md-imrphenom-s41.toml" in result.stdout
    assert "config/banks/uniform-imrphenom-s51.toml" in result.stdout
    assert "config/populations/madau-dickinson.yaml" in result.stdout
    assert "config/populations/uniform-redshift.yaml" in result.stdout
    assert "outputs/banks/md-imrphenom-s41.h5" in result.stdout
    assert "outputs/banks/uniform-imrphenom-s51.h5" in result.stdout
    assert "astrogwb-generate-bank" in result.stdout
    # The population intermediate and its merge rule are both gone.
    assert "outputs/populations/" not in result.stdout
    assert "outputs/population-configs/" not in result.stdout
    # Mixing moved to CatalogSource.compose; generation is single-component now.
    assert "--uniform-mixing-fraction" not in result.stdout


def test_the_snakefile_no_longer_needs_ancient() -> None:
    """Per-run configs restore real change tracking.

    ``ancient()`` existed only because one rule emitted all 26 configs, so any
    edit invalidated every chain. It also meant a config change never
    retriggered sampling at all.
    """
    assert "ancient(" not in SNAKEFILE.read_text()


def test_banks_target_builds_all_4_banks() -> None:
    result = _snakemake(
        "--snakefile",
        str(SNAKEFILE),
        "--allowed-rules",
        *BANK_RULES,
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "banks",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule waveform_bank:") == 4
    assert "rule population_config:" not in result.stdout
    for name in (
        "md-imrphenom-s41",
        "md-imrphenom-s42",
        "md-taylorf2-s41",
        "uniform-imrphenom-s51",
    ):
        assert f"outputs/banks/{name}.h5" in result.stdout


def test_plot_cosmological_parameters_expands_all_chains_and_figures(
    tmp_path: Path,
) -> None:
    banks = _banks(tmp_path, "md-imrphenom-s42.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "plot_cosmological_parameters",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    # Eight chains and no config-assembly jobs: the layers are inputs of
    # run_mcmc, not outputs of a rule of their own.
    assert "rule assemble_config:" not in result.stdout
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
    banks = _banks(tmp_path, "md-imrphenom-s42.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "run_experiment_cosmological_parameters",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 8
    assert "rule plot_cosmological_parameters:" not in result.stdout


def test_run_mcmc_is_handed_its_layers_on_argv(tmp_path: Path) -> None:
    banks = _banks(tmp_path, "md-imrphenom-s42.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "4",
        "outputs/chains/cosmological-parameters/H0-Omega_m.nc",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    # No intermediate artifact: the layers are the rule's inputs *and* what it
    # passes on argv, so the dependency edges and the data path are one list.
    assert "astrogwb-assemble-config" not in result.stdout
    assert "outputs/configs/" not in result.stdout
    assert "rule assemble_config:" not in result.stdout

    layers = [
        str(path.relative_to(PAPER_ROOT))
        for path in run_config_paths(
            "cosmological-parameters", "H0-Omega_m", root=PAPER_ROOT
        )
    ]
    # Three layers, all declared, so any of them retriggers this run alone.
    assert "config/analysis/base/parameters.toml" in layers
    assert "config/analysis/runs/cosmological-parameters/_base.toml" in layers
    assert "config/analysis/runs/cosmological-parameters/H0-Omega_m.toml" in layers
    for layer in layers:
        assert f"--config {layer}" in result.stdout
        assert any(layer in line for line in _rule_inputs(result.stdout))
    # Repeated, not space-joined: argparse's append action takes one path each.
    assert result.stdout.count("--config config/analysis/") >= len(layers)

    assert "--bank" in result.stdout
    assert f"md-imrphenom-s41={banks / 'md-imrphenom-s41.h5'}" in result.stdout
    assert f"md-imrphenom-s42={banks / 'md-imrphenom-s42.h5'}" in result.stdout


def test_variable_catalog_size_shares_one_bank_across_all_sizes(
    tmp_path: Path,
) -> None:
    """n8192/n16384/n32768 are prefixes of the same bank -- one file, not three."""
    banks = _banks(tmp_path, "md-imrphenom-s42.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "8",
        "run_experiment_variable_catalog_size",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 3
    assert result.stdout.count(str(banks / "md-imrphenom-s42.h5")) >= 3


def test_variable_proposal_guard_shares_md_and_uniform_banks(
    tmp_path: Path,
) -> None:
    banks = _banks(tmp_path, "md-imrphenom-s42.h5", "uniform-imrphenom-s51.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "run_experiment_variable_proposal_guard",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 3
    for name in ("md-imrphenom-s42.h5", "uniform-imrphenom-s51.h5"):
        assert str(banks / name) in result.stdout


def test_waveform_approximant_uses_imr_and_taylorf2_banks(
    tmp_path: Path,
) -> None:
    banks = _banks(tmp_path, "md-taylorf2-s41.h5")
    imr = banks / "md-imrphenom-s41.h5"
    taylorf2 = banks / "md-taylorf2-s41.h5"

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "8",
        "run_experiment_waveform_approximant",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 2
    # IMRPhenom run: injection and proposal share one bank (one job's worth);
    # TaylorF2 run: injection still needs the IMR bank -- two jobs use it.
    assert result.stdout.count(f"md-imrphenom-s41={imr}") == 2
    assert result.stdout.count(f"md-taylorf2-s41={taylorf2}") == 1


def test_missing_bank_does_not_acquire_a_producer(tmp_path: Path) -> None:
    banks = tmp_path / "missing-banks"

    result = _mcmc(
        "--dry-run",
        "--cores",
        "4",
        "run_experiment_cosmological_parameters",
        "--config",
        f"banks_dir={banks}",
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "MissingInputException" in output
    assert "waveform_bank" not in output


def test_unified_workflow_exposes_explicit_experiment_targets() -> None:
    result = _snakemake("--snakefile", str(SNAKEFILE), "--list-rules")

    assert result.returncode == 0, result.stderr
    rules = _rule_names(result.stdout)
    assert {
        "run_experiment_cosmological_parameters",
        "run_experiment_modified_propagation",
        "run_experiment_astrophysical_parameters",
        "run_experiment_variable_catalog_size",
        "run_experiment_variable_proposal_guard",
        "run_experiment_waveform_approximant",
        "banks",
        "plot_cosmological_parameters",
        "amplitude_toy",
        "fiducial_spectrum",
        "importance_weights_grid",
        "validate",
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
        # The old per-composition catalog rules are gone entirely.
        "catalogs",
        "population",
        "waveform_catalog",
        # The population left the DAG: it was a temp() node with one consumer.
        "population_config",
        "population_bank",
        # The assembled-config artifact and its aggregate target are gone: each
        # entrypoint merges the layer files the rule hands it.
        "assemble_config",
        "configs",
    }.isdisjoint(rules)


def test_experiments_target_builds_all_26_chains(tmp_path: Path) -> None:
    banks = _banks(
        tmp_path,
        "md-imrphenom-s42.h5",
        "md-taylorf2-s41.h5",
        "uniform-imrphenom-s51.h5",
    )

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "experiments",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    assert "rule assemble_config:" not in result.stdout
    assert result.stdout.count("rule run_mcmc:") == 26
    # `experiments` is chains-only now; figures are opt-in via the plot rules.
    assert "rule plot_cosmological_parameters:" not in result.stdout
    assert "rule plot_modified_propagation:" not in result.stdout


def test_plot_cosmological_parameters_passes_all_paths_not_labels(
    tmp_path: Path,
) -> None:
    banks = _banks(tmp_path, "md-imrphenom-s42.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "8",
        "plot_cosmological_parameters",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    assert "rule plot_cosmological_parameters:" in result.stdout
    # The figure is handed the layer files of a run it actually plots, and the
    # rule declares those same files -- so the edge that retriggers the figure
    # and the data path that fills it in are one list, not two.
    for layer in FIGURE_CONFIG_LAYERS:
        assert f"--config {layer}" in result.stdout
        assert any(layer in line for line in _rule_inputs(result.stdout))
    assert "--base-config" not in result.stdout
    assert "outputs/configs/" not in result.stdout
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
    # The script hard-codes its own labels, so no LaTeX crosses the shell
    # boundary. Run *names* do: detectors are per-run and are read back from
    # each run's own config.
    assert "--figure-config" not in result.stdout
    assert "--prior-labels" not in result.stdout
    assert r"\mathcal" not in result.stdout
    for run in DETECTOR_NETWORK_RUNS:
        assert f"--network-run cosmological-parameters/{run}" in result.stdout


def test_network_run_flags_follow_the_legend_order(tmp_path: Path) -> None:
    # Declaration order drives chain order, legend order, and colour
    # assignment. `resolve_networks` rejects a mis-ordered list, so this pins
    # that the workflow emits the order it expects rather than relying on the
    # figure to still render.
    banks = _banks(tmp_path, "md-imrphenom-s42.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "8",
        "plot_cosmological_parameters",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    command = result.stdout[result.stdout.index("--network-run ") :]
    positions = [
        command.index(f"--network-run cosmological-parameters/{run}")
        for run in DETECTOR_NETWORK_RUNS
    ]
    assert positions == sorted(positions)


def test_standalone_figures_receive_config_paths(
    tmp_path: Path,
) -> None:
    banks = _banks(tmp_path, "md-imrphenom-s42.h5")

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
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    for script in (
        "scripts/amplitude_toy_model.py",
        "scripts/fiducial_spectrum.py",
        "scripts/importance_weights_grid.py",
    ):
        assert script in result.stdout
    # Each standalone script is handed config *layers*, never fiducials and
    # analysis bounds reconstructed into flags, and never an assembled config.
    assert "--base-config" not in result.stdout
    assert "--figure-config" not in result.stdout
    assert "outputs/configs/" not in result.stdout
    for layer in FIGURE_CONFIG_LAYERS:
        assert sum(layer in line for line in _rule_inputs(result.stdout)) == 3
        assert result.stdout.count(f"--config {layer}") == 3
    for flag in ("--observation-time", "--f-min", "--h0", "--omega-gw-min"):
        assert flag not in result.stdout
    # fiducial_spectrum borrows the cosmological-parameters networks and reads
    # no chains; it still declares their TOMLs, so editing one retriggers it.
    assert result.stdout.count("--network-run cosmological-parameters/") == len(
        DETECTOR_NETWORK_RUNS
    )


def test_figure_path_is_a_valid_snakemake_target(
    tmp_path: Path,
) -> None:
    banks = _banks(tmp_path, "md-imrphenom-s42.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--cores",
        "8",
        "outputs/figures/cosmological-parameters/H0-by-detector.pdf",
        "--config",
        f"banks_dir={banks}",
    )

    assert result.returncode == 0, result.stderr
    assert "rule plot_cosmological_parameters:" in result.stdout
    assert result.stdout.count("rule run_mcmc:") == 8


def test_figure_rule_preserves_declared_chain_order(tmp_path: Path) -> None:
    banks = _banks(tmp_path, "md-imrphenom-s42.h5")

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "8",
        "plot_cosmological_parameters",
        "--config",
        f"banks_dir={banks}",
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
