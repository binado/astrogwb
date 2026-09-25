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
CATALOG_RULES = ("merge_catalog_config", "waveform_catalog", "catalogs")
MCMC_RULES = (
    "validate",
    "run_mcmc",
    "plot_cosmological_parameters",
    "plot_modified_propagation",
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


#: Every catalog name the 27 runs draw on, minus the default injection one
#: that ``_catalogs`` always creates.
NON_INJECTION_CATALOGS = (
    "md-imrphenom-s42-n8192.h5",
    "md-imrphenom-s42-n16384.h5",
    "md-imrphenom-s42-n32768.h5",
    "md-taylorf2-s41-n32768.h5",
    "md-uniform-imrphenom-s61-n16384-eps1e-1.h5",
    "md-uniform-imrphenom-s62-n16384-eps1e-2.h5",
    "md-uniform-imrphenom-s63-n16384-eps1e-3.h5",
    "md-delayed-imrphenom-s71-n32768.h5",
)


def _all_catalogs(tmp_path: Path) -> Path:
    """A fake catalogs directory holding all nine."""
    return _catalogs(tmp_path, *NON_INJECTION_CATALOGS)


def _catalogs(tmp_path: Path, *names: str) -> Path:
    """A fake catalogs directory: the shared injection catalog plus ``names``."""
    directory = tmp_path / "catalogs"
    directory.mkdir(exist_ok=True)
    (directory / "md-imrphenom-s41-n32768.h5").touch()
    for name in names:
        (directory / name).touch()
    return directory


def test_catalog_rule_reads_its_config_layers_directly() -> None:
    """Two rules per catalog: fold the layers once, then draw from the result.

    The population used to be a separate graph file declared as an extra
    input. It is a registered model named by `config/population.json` now, and
    the hyperparameters come from `config/fiducials.json`, so the layer list
    *is* the dependency edge. That edge now sits on `merge_catalog_config`,
    and `waveform_catalog` inherits it through the merged file -- which is
    what lets the fold happen once instead of once per flag.
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
    # The shared waveform layer is an input of both, so editing it rebuilds both.
    assert result.stdout.count("config/waveform.json") >= 2
    assert "config/catalogs/md-imrphenom-s41-n32768.json" in result.stdout
    assert (
        "config/catalogs/md-uniform-imrphenom-s61-n16384-eps1e-1.json" in result.stdout
    )
    # Editing either shared declaration rebuilds every catalog. fiducials.json
    # is a run layer too, so the hyperparameters a catalog is drawn at and the
    # ones a run initializes at cannot drift.
    assert result.stdout.count("config/population.json") >= 2
    assert result.stdout.count("config/fiducials.json") >= 2
    assert "outputs/catalogs/md-imrphenom-s41-n32768.h5" in result.stdout
    assert (
        "outputs/catalogs/md-uniform-imrphenom-s61-n16384-eps1e-1.h5" in result.stdout
    )
    assert "python scripts/generate_catalog.py" in result.stdout
    assert any(
        "scripts/generate_catalog.py" in line for line in _rule_inputs(result.stdout)
    )
    # The fold runs once per catalog, over exactly the declared layers, into
    # the merged file. This is the assertion that would fail if the merge
    # migrated back into the flags and started re-folding once each.
    layers = (
        "config/waveform.json config/population.json config/fiducials.json "
        "config/catalogs/md-imrphenom-s41-n32768.json"
    )
    merged = "outputs/catalogs/md-imrphenom-s41-n32768.merged.json"
    merge = "reduce .[] as $layer ({}; . * $layer)"
    assert f"jq -s '{merge}' {layers} > {merged}" in result.stdout
    assert result.stdout.count(f"jq -s '{merge}'") == 2, "one fold per catalog"

    # The generator reads keys out of that one file, never the layer tree.
    for flag, compact in (
        ("--population", True),
        ("--fiducials", True),
        ("--waveform", True),
        ("--seed", False),
        ("--num-samples", False),
    ):
        key = flag.removeprefix("--").replace("-", "_")
        jq = "jq -c" if compact else "jq -r"
        assert f'{flag} "$({jq} .{key} {merged})"' in result.stdout, flag
    assert "--config" not in result.stdout
    # The old base/ and defs/ split is gone: one flat directory of defs.
    assert "config/catalogs/base/" not in result.stdout
    assert "config/catalogs/defs/" not in result.stdout
    # The population intermediate, its merge rule, and the graph configs the
    # rule used to declare are all gone.
    assert "outputs/populations/" not in result.stdout
    assert "outputs/population-configs/" not in result.stdout
    assert "config/populations/" not in result.stdout
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


def test_catalogs_target_builds_all_9_catalogs() -> None:
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
    assert result.stdout.count("rule waveform_catalog:") == 9
    assert "rule population_config:" not in result.stdout


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
    assert result.stdout.count("rule run_mcmc:") == 27
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
        "importance_weights_grid",
        "--config",
        f"catalogs_dir={catalogs}",
    )

    assert result.returncode == 0, result.stderr
    assert "scripts/importance_weights_grid.py" in result.stdout
    # Each standalone script is handed config *layers*, never fiducials and
    # analysis bounds reconstructed into flags, and never an assembled config.
    assert "--base-config" not in result.stdout
    assert "--figure-config" not in result.stdout
    assert "outputs/configs/" not in result.stdout
    # The standalone rule receives each shared layer as both an input and a
    # repeated --config flag.
    for layer in FIGURE_CONFIG_LAYERS:
        assert sum(layer in line for line in _rule_inputs(result.stdout)) == 1
        assert result.stdout.count(f"--config {layer}") == 1
    for flag in ("--observation-time", "--f-min", "--h0", "--omega-gw-min"):
        assert flag not in result.stdout


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
