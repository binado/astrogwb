"""Workflow-shape tests: what the DAG builds, and what each rule is handed.

Every dry run happens in a scratch directory holding symlinks to the committed
inputs (``Snakefile``, ``config/``, ``scripts/``) and nothing else. Running them
against the real checkout instead would make them depend on whichever chains
and catalogs a developer happens to have built -- and existing chains are
``protected()``, so a ``--forceall`` dry run against them fails outright.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from repo import REPO_ROOT

from astrogwb.paper.config.runs import run_config_paths
from astrogwb.paper.plotting import DETECTOR_NETWORK_RUNS

PAPER_ROOT = REPO_ROOT
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
CATALOG_RULES = ("waveform_catalog", "catalogs")
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


#: Every catalog name the 26 runs draw on, minus the shared injection one
#: that ``_catalogs`` always creates.
NON_INJECTION_CATALOGS = (
    "md-imrphenom-s42-n8192.h5",
    "md-imrphenom-s42-n16384.h5",
    "md-imrphenom-s42-n32768.h5",
    "md-taylorf2-s41-n32768.h5",
    "md-uniform-imrphenom-s61-n16384-eps1e-1.h5",
    "md-uniform-imrphenom-s62-n16384-eps1e-2.h5",
    "md-uniform-imrphenom-s63-n16384-eps1e-3.h5",
)


def _all_catalogs(tmp_path: Path) -> Path:
    """A fake catalogs directory holding all eight."""
    return _catalogs(tmp_path, *NON_INJECTION_CATALOGS)


def _catalogs(tmp_path: Path, *names: str) -> Path:
    """A fake catalogs directory: the shared injection catalog plus ``names``."""
    directory = tmp_path / "catalogs"
    directory.mkdir(exist_ok=True)
    (directory / "md-imrphenom-s41-n32768.h5").touch()
    for name in names:
        (directory / name).touch()
    return directory


def test_catalog_rule_reads_its_layers_and_populations_directly() -> None:
    """One rule per catalog; the population is not a workflow node.

    A mixture declares both graphs, so editing either retriggers only the
    catalogs that draw from it.
    """
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
        "outputs/catalogs/md-imrphenom-s41-n32768.h5",
        "outputs/catalogs/md-uniform-imrphenom-s61-n16384-eps1e-1.h5",
    )

    assert result.returncode == 0, result.stderr
    # The shared base layer is an input of both, so editing it rebuilds both.
    assert result.stdout.count("config/catalogs/base/waveform.toml") >= 2
    assert "config/catalogs/defs/md-imrphenom-s41-n32768.toml" in result.stdout
    assert (
        "config/catalogs/defs/md-uniform-imrphenom-s61-n16384-eps1e-1.toml"
        in result.stdout
    )
    assert "config/populations/madau-dickinson.yaml" in result.stdout
    assert "config/populations/uniform-redshift.yaml" in result.stdout
    assert "outputs/catalogs/md-imrphenom-s41-n32768.h5" in result.stdout
    assert (
        "outputs/catalogs/md-uniform-imrphenom-s61-n16384-eps1e-1.h5" in result.stdout
    )
    assert "python scripts/generate_catalog.py" in result.stdout
    assert any(
        "scripts/generate_catalog.py" in line for line in _rule_inputs(result.stdout)
    )
    # Layers reach the script as repeated flags, never space-joined into one.
    assert (
        "--config config/catalogs/base/waveform.toml "
        "--config config/catalogs/defs/md-imrphenom-s41-n32768.toml"
    ) in result.stdout
    # The population intermediate and its merge rule are both gone.
    assert "outputs/populations/" not in result.stdout
    assert "outputs/population-configs/" not in result.stdout
    # The bank/catalog split is gone: no separate bank tree, no bank rule.
    assert "outputs/banks/" not in result.stdout
    assert "config/banks/" not in result.stdout


def test_the_snakefile_no_longer_needs_ancient() -> None:
    """Per-run configs restore real change tracking.

    ``ancient()`` existed only because one rule emitted all 26 configs, so any
    edit invalidated every chain. It also meant a config change never
    retriggered sampling at all.
    """
    assert "ancient(" not in SNAKEFILE.read_text()


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
    assert result.stdout.count("rule waveform_catalog:") == 8
    assert "rule population_config:" not in result.stdout
    for name in (
        "md-imrphenom-s41-n32768",
        "md-imrphenom-s42-n8192",
        "md-imrphenom-s42-n16384",
        "md-imrphenom-s42-n32768",
        "md-taylorf2-s41-n32768",
        "md-uniform-imrphenom-s61-n16384-eps1e-1",
        "md-uniform-imrphenom-s62-n16384-eps1e-2",
        "md-uniform-imrphenom-s63-n16384-eps1e-3",
    ):
        assert f"outputs/catalogs/{name}.h5" in result.stdout


def test_plot_cosmological_parameters_expands_all_chains_and_figures(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "md-imrphenom-s42-n16384.h5")

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
    catalogs = _catalogs(tmp_path, "md-imrphenom-s42-n16384.h5")

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


def test_run_mcmc_is_handed_its_layers_on_argv(tmp_path: Path) -> None:
    catalogs = _catalogs(tmp_path, "md-imrphenom-s42-n16384.h5")

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
    # No intermediate artifact: the layers are the rule's inputs *and* what it
    # passes on argv, so the dependency edges and the data path are one list.
    assert "astrogwb-assemble-config" not in result.stdout
    assert "outputs/configs/" not in result.stdout
    assert "rule assemble_config:" not in result.stdout
    assert "python scripts/run_mcmc.py" in result.stdout
    assert any("scripts/run_mcmc.py" in line for line in _rule_inputs(result.stdout))

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

    # Roles are fixed, so the two files arrive as named flags with no
    # name-to-path mapping to parse.
    assert (
        f"--injection-catalog {catalogs / 'md-imrphenom-s41-n32768.h5'}"
        in result.stdout
    )
    assert (
        f"--proposal-catalog {catalogs / 'md-imrphenom-s42-n16384.h5'}" in result.stdout
    )


def test_variable_catalog_size_names_one_catalog_per_size(
    tmp_path: Path,
) -> None:
    """One file per size now: the three are nested draws, not bank prefixes."""
    catalogs = _catalogs(
        tmp_path,
        "md-imrphenom-s42-n8192.h5",
        "md-imrphenom-s42-n16384.h5",
        "md-imrphenom-s42-n32768.h5",
    )

    result = _mcmc(
        "--dry-run",
        "--forceall",
        "--printshellcmds",
        "--cores",
        "8",
        "run_experiment_variable_catalog_size",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("rule run_mcmc:") == 3
    for name in (
        "md-imrphenom-s42-n8192.h5",
        "md-imrphenom-s42-n16384.h5",
        "md-imrphenom-s42-n32768.h5",
    ):
        assert str(catalogs / name) in result.stdout
    # All three share the one injection catalog.
    assert result.stdout.count(str(catalogs / "md-imrphenom-s41-n32768.h5")) >= 3


def test_variable_proposal_guard_names_one_catalog_per_eps(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(
        tmp_path,
        "md-uniform-imrphenom-s61-n16384-eps1e-1.h5",
        "md-uniform-imrphenom-s62-n16384-eps1e-2.h5",
        "md-uniform-imrphenom-s63-n16384-eps1e-3.h5",
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
        "md-uniform-imrphenom-s61-n16384-eps1e-1.h5",
        "md-uniform-imrphenom-s62-n16384-eps1e-2.h5",
        "md-uniform-imrphenom-s63-n16384-eps1e-3.h5",
    ):
        assert str(catalogs / name) in result.stdout


def test_waveform_approximant_uses_imr_and_taylorf2_catalogs(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "md-taylorf2-s41-n32768.h5")
    imr = catalogs / "md-imrphenom-s41-n32768.h5"
    taylorf2 = catalogs / "md-taylorf2-s41-n32768.h5"

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
    # The IMRPhenom run's proposal *is* the injection catalog -- the same file
    # in both roles, which is what collapsed nine specs into eight files.
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
    assert "waveform_catalog" not in output


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
        "catalogs",
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
        "population",
        # The bank/catalog split is gone: one artifact kind, one rule.
        "banks",
        "waveform_bank",
        # The population left the DAG: it was a temp() node with one consumer.
        "population_config",
        "population_bank",
        # The assembled-config artifact and its aggregate target are gone: each
        # entrypoint merges the layer files the rule hands it.
        "assemble_config",
        "configs",
    }.isdisjoint(rules)


def test_experiments_target_builds_all_26_chains(tmp_path: Path) -> None:
    catalogs = _all_catalogs(tmp_path)

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
    assert "rule assemble_config:" not in result.stdout
    assert result.stdout.count("rule run_mcmc:") == 26
    # `experiments` is chains-only now; figures are opt-in via the plot rules.
    assert "rule plot_cosmological_parameters:" not in result.stdout
    assert "rule plot_modified_propagation:" not in result.stdout


def test_plot_cosmological_parameters_passes_all_paths_not_labels(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "md-imrphenom-s42-n16384.h5")

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
    catalogs = _catalogs(tmp_path, "md-imrphenom-s42-n16384.h5")

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
    command = result.stdout[result.stdout.index("--network-run ") :]
    positions = [
        command.index(f"--network-run cosmological-parameters/{run}")
        for run in DETECTOR_NETWORK_RUNS
    ]
    assert positions == sorted(positions)


def test_standalone_figures_receive_config_paths(
    tmp_path: Path,
) -> None:
    catalogs = _catalogs(tmp_path, "md-imrphenom-s42-n16384.h5")

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
    catalogs = _catalogs(tmp_path, "md-imrphenom-s42-n16384.h5")

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
    catalogs = _catalogs(tmp_path, "md-imrphenom-s42-n16384.h5")

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
