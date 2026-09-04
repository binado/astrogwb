"""Discover, locate, and merge MCMC run configs from the ``config/analysis/`` tree.

Filenames are the mapping. ``config/analysis/runs/<experiment>/<run>.toml``
samples into ``outputs/chains/<experiment>/<run>.nc``; no inventory file
translates between the two. That convention is what let the previous
``inputs/experiments.yaml`` registry -- and the ten ``Snakefile`` helpers that
read it -- go away.

A run config is three layers merged in order:

1. ``config/analysis/base/*.toml`` -- settings every run shares.
2. ``config/analysis/runs/<experiment>/_base.toml`` -- the experiment override.
3. ``config/analysis/runs/<experiment>/<run>.toml`` -- the run override.

``_base.toml`` is required in every experiment directory rather than optional:
a conditional Snakemake input complicates the DAG for no gain.

There is no assembled-config artifact. Every entrypoint is handed its layer
files on argv (see :func:`add_config_arguments`) and merges them in process;
:func:`assemble_run` is the convenience wrapper for the notebooks and for the
validation gate, which address a run by name rather than by path.

**stdlib only, and deliberately so.** The ``Snakefile`` imports this module to
build the DAG, so it must not reach pydantic, JAX, or ``astrogwb``: a
validation error in any one run would otherwise break DAG construction for
every target, and every ``--dry-run`` would pay for a JAX import. Bank
*validation* lives in :mod:`astrogwb.paper.config.banks` for that reason.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrogwb.paper.utils import deep_merge, load_mapping

if TYPE_CHECKING:
    from astrogwb.paper.plotting import Network

logger = logging.getLogger(__name__)

#: Relative to the working directory, which for the workflow and every script
#: is the repository root. Library code names no absolute path and does not go
#: looking for a checkout: the caller's cwd is the answer.
ANALYSIS_DIR = Path("config/analysis")
BASE_DIR = ANALYSIS_DIR / "base"
RUNS_DIR = ANALYSIS_DIR / "runs"
EXPERIMENT_BASE = "_base.toml"

CHAINS_ROOT = Path("outputs/chains")


def merge_run_overlay(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge a run overlay, replacing named prior tables wholesale.

    ``deep_merge`` key-merges nested mappings, which leaves stale ``low`` /
    ``high`` behind when a uniform prior is replaced by a normal one. Each
    ``[priors.<param>]`` table in ``override`` replaces the base spec instead.
    """
    overlay_priors = override.get("priors")
    merged = deep_merge(
        base, {key: value for key, value in override.items() if key != "priors"}
    )
    if not isinstance(overlay_priors, Mapping):
        return merged
    priors = dict(merged.get("priors") or {})
    for name, spec in overlay_priors.items():
        priors[name] = dict(spec) if isinstance(spec, Mapping) else spec
    merged["priors"] = priors
    return merged


def merge_config_layers(paths: Sequence[Path]) -> dict[str, Any]:
    """Fold run-config layer files into one raw mapping, in the order given.

    This is a *run-config* parser, not generic config infrastructure: it folds
    :func:`merge_run_overlay`, whose prior-replacement rule is domain-specific.
    A run that swaps a uniform prior for a normal one must not inherit the
    uniform's ``low`` / ``high``, and a plain deep merge would leave them
    behind.

    Order is the caller's responsibility and it is not recoverable from the
    result, so it is logged. Applying the same fold to the ``base/`` files is a
    no-op difference from a plain deep merge -- they partition disjoint
    top-level keys -- so one function serves every layer.
    """
    if not paths:
        raise ValueError("no config layers given")
    merged: dict[str, Any] = {}
    for path in paths:
        merged = merge_run_overlay(merged, load_mapping(path))
    return merged


def discover_runs(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Return ``{experiment: (run, ...)}`` by globbing the runs tree.

    ``_base.toml`` is the experiment override, not a run, so it is excluded.
    """
    resolved = root or Path()
    runs_dir = resolved / RUNS_DIR
    experiments = tuple(
        sorted(path.name for path in runs_dir.iterdir() if path.is_dir())
    )
    if not experiments:
        raise ValueError(f"{runs_dir} declares no experiments")
    runs: dict[str, tuple[str, ...]] = {}
    for experiment in experiments:
        directory = resolved / RUNS_DIR / experiment
        if not (directory / EXPERIMENT_BASE).is_file():
            raise ValueError(f"{directory} is missing a required {EXPERIMENT_BASE}")
        names = tuple(
            sorted(
                path.stem
                for path in directory.glob("*.toml")
                if path.name != EXPERIMENT_BASE
            )
        )
        if not names:
            raise ValueError(f"{directory} declares no runs")
        runs[experiment] = names
    return runs


def base_config_paths(root: Path | None = None) -> tuple[Path, ...]:
    """Every shared ``config/analysis/base/*.toml`` layer, in merge order.

    Sorted for determinism only: the base files partition disjoint top-level
    keys, so the order does not change the outcome.
    """
    directory = (root or Path()) / BASE_DIR
    paths = tuple(sorted(directory.glob("*.toml")))
    if not paths:
        raise ValueError(f"{directory} declares no base config files")
    return paths


def run_config_paths(
    experiment: str, run: str, *, root: Path | None = None
) -> tuple[Path, ...]:
    """The ordered layer files that make up one run's config.

    The ``Snakefile`` declares exactly these as ``run_mcmc``'s config inputs
    and passes them back on argv, so the dependency edges and the data path are
    the same list.
    """
    resolved = root or Path()
    directory = resolved / RUNS_DIR / experiment
    run_path = directory / f"{run}.toml"
    if not run_path.is_file():
        raise ValueError(f"unknown run {experiment}/{run}: {run_path} does not exist")
    experiment_base = directory / EXPERIMENT_BASE
    if not experiment_base.is_file():
        raise ValueError(f"{directory} is missing a required {EXPERIMENT_BASE}")
    return (*base_config_paths(resolved), experiment_base, run_path)


def load_base(root: Path | None = None) -> dict[str, Any]:
    """Merge every ``config/analysis/base/*.toml`` into one mapping."""
    return merge_config_layers(base_config_paths(root))


def assemble_run(
    experiment: str, run: str, *, root: Path | None = None
) -> dict[str, Any]:
    """Merge one run's three layers into a raw config, addressing it by name.

    The convenience wrapper for callers that hold ``(experiment, run)`` rather
    than a list of paths: the notebooks, the validation gate, and the figure
    scripts resolving a ``--network-run``.
    """
    return merge_config_layers(run_config_paths(experiment, run, root=root))


def catalog_bank_names(raw: Mapping[str, Any]) -> list[str]:
    """Sorted distinct bank names named by a raw config's ``[catalog]`` block."""
    catalog = raw.get("catalog")
    if not isinstance(catalog, Mapping):
        raise TypeError("run config must define a [catalog] table")
    names: set[str] = set()
    for role in ("injection", "proposal"):
        spec = catalog.get(role)
        if not isinstance(spec, Mapping):
            raise TypeError(f"run config must define a [catalog.{role}] table")
        for key in ("md_bank", "uniform_bank"):
            value = spec.get(key)
            if value:
                names.add(str(value))
    return sorted(names)


def resolve_bank_names(
    experiment: str, run: str, *, root: Path | None = None
) -> list[str]:
    """Every distinct bank a run's injection and proposal catalogs draw from.

    The ``Snakefile`` calls this to declare ``run_mcmc``'s bank inputs, which is
    why it works off the merged mapping rather than a validated
    :class:`~astrogwb.paper.config.mcmc.RunConfig`: the DAG must be buildable
    without paying for full validation of all 26 runs.
    """
    return catalog_bank_names(assemble_run(experiment, run, root=root))


def resolve_networks(
    references: Sequence[tuple[str, str]],
    networks: Sequence[tuple[str, str]],
    *,
    root: Path | None = None,
) -> tuple[Network, ...]:
    """Attach detectors to each ``(run, label)`` pair from that run's own config.

    ``references`` are the ``--network-run <experiment>/<run>`` values a figure
    rule passed; ``networks`` is the ordered ``(run, label)`` legend, normally
    :data:`astrogwb.paper.plotting.DETECTOR_NETWORKS`.

    The two lists are matched *positionally* and checked, because declaration
    order drives chain order, legend order, and the color/linestyle assignment
    in the detector-comparison figures -- a mis-ordered flag list would render
    a perfectly good figure with the wrong labels on the wrong curves. The
    experiment is taken from the references rather than hard-coded, and all of
    them must name the same one: comparing networks across experiments would
    silently mix two different models.
    """
    # Imported here, not at module scope: `Network` lives in `plotting`, which
    # imports matplotlib, and this module must stay importable by the Snakefile
    # without it. Only figure scripts call this.
    from astrogwb.paper.plotting import Network

    if not networks:
        raise ValueError("figure declares no detector networks")
    expected = [name for name, _ in networks]
    duplicates = sorted({run for run in expected if expected.count(run) > 1})
    if duplicates:
        raise ValueError("duplicate detector network(s): " + ", ".join(duplicates))

    if len(references) != len(networks):
        raise ValueError(
            f"--network-run was given {len(references)} runs but the figure "
            f"legend declares {len(networks)}"
        )
    experiments = {experiment for experiment, _ in references}
    if len(experiments) != 1:
        raise ValueError(
            "every --network-run must name the same experiment, got: "
            + ", ".join(sorted(experiments))
        )
    experiment = next(iter(experiments))
    given = [run for _, run in references]
    if given != expected:
        raise ValueError(
            "--network-run order must match the figure legend order.\n"
            f"  given:    {', '.join(given)}\n"
            f"  expected: {', '.join(expected)}"
        )

    resolved: list[Network] = []
    for name, label in networks:
        merged = assemble_run(experiment, name, root=root)
        analysis = merged.get("analysis") or {}
        detectors = analysis.get("detectors")
        if not detectors:
            raise ValueError(f"{experiment}/{name} declares no analysis.detectors")
        resolved.append(Network(name, label, tuple(detectors)))
    return tuple(resolved)


def add_network_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the repeated ``--network-run`` flag the network figures take.

    Three layers times eight runs of ``--config`` is unworkable, so the network
    figures are handed run *names* and re-derive the layer paths themselves.
    The workflow still declares those TOMLs as ``input:``, so the edges are
    real.
    """
    parser.add_argument(
        "--network-run",
        dest="network_runs",
        action="append",
        type=parse_run_reference,
        required=True,
        metavar="EXPERIMENT/RUN",
        help=(
            "One detector-network run, as <experiment>/<run>; repeat once per "
            "network, in the figure's legend order."
        ),
    )


def parse_run_reference(value: str) -> tuple[str, str]:
    """Parse an ``<experiment>/<run>`` CLI reference into its two parts."""
    experiment, sep, run = value.partition("/")
    if not sep or not experiment or not run or "/" in run:
        raise argparse.ArgumentTypeError(f"expected <experiment>/<run>, got {value!r}")
    return experiment, run


def run_target(experiment: str) -> str:
    """Return the Snakemake target that samples every run in an experiment."""
    return f"run_experiment_{experiment.replace('-', '_')}"


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the repeated ``--config`` layer flag shared by every entrypoint.

    Mirrors :func:`astrogwb.paper.runtime.add_runtime_arguments`: one
    definition, so all entrypoints spell the flag the same way.
    """
    parser.add_argument(
        "--config",
        dest="config",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "One run-config layer file, in merge order; repeat once per layer "
            "(base/*.toml, then the experiment _base.toml, then the run)."
        ),
    )


def load_merged_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the ``--config`` layers an entrypoint was handed, order preserved.

    Merge order is the caller's to get right now that no single function owns
    it, and a wrong-but-valid order fails silently, so the resolved order is
    logged before the merge and recorded next to every chain by
    :mod:`astrogwb.paper.cli.run_mcmc`.
    """
    paths: list[Path] = list(args.config)
    logger.info(
        "Config layers (merge order): %s", " -> ".join(str(path) for path in paths)
    )
    return merge_config_layers(paths)
