"""Discover, locate, and merge MCMC run configs from the ``config/analysis/`` tree.

Filenames are the mapping. ``config/analysis/runs/<experiment>/<run>.toml``
samples into ``outputs/chains/<experiment>/<run>.nc``; no inventory file
translates between the two. That convention is what let the previous
``inputs/experiments.yaml`` registry -- and the ten ``Snakefile`` helpers that
read it -- go away.

A run config is four layers merged in order:

0. ``config/{fiducials,priors,networks}.json`` -- the shared scientific values,
   which the notebooks and figure scripts also read directly through
   :mod:`astrogwb.paper.config`. JSON so that ``jq`` can read them without
   importing the package.
1. ``config/analysis/base/*.toml`` -- the remaining settings every run shares.
2. ``config/analysis/runs/<experiment>/_base.toml`` -- the experiment override.
3. ``config/analysis/runs/<experiment>/<run>.toml`` -- the run override.

Layers 0 and 1 together are :func:`base_config_paths`, so a caller that wants
"everything shared" asks for it once.

``_base.toml`` is required in every experiment directory rather than optional:
a conditional Snakemake input complicates the DAG for no gain.

There is no assembled-config artifact. Every entrypoint is handed its layer
files on argv (see :func:`add_config_arguments`) and merges them in process;
:func:`assemble_run` is the convenience wrapper for the notebooks and for the
validation gate, which address a run by name rather than by path.

**stdlib only, and deliberately so.** The ``Snakefile`` imports this module to
build the DAG, so it must not reach pydantic, JAX, or ``astrogwb``: a
validation error in any one run would otherwise break DAG construction for
every target, and every ``--dry-run`` would pay for a JAX import. Catalog
*validation* lives in :mod:`astrogwb.paper.config.catalogs` for that reason.
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
CONFIG_DIR = Path("config")

#: Layer 0: the values shared by every run *and* read directly by the notebooks
#: and figure scripts through `astrogwb.paper.config`. They are JSON so that
#: `jq` can read them without importing the package, and they are ordinary
#: merge layers -- each is a single-key object -- so nothing here special-cases
#: them.
FIDUCIALS_PATH = CONFIG_DIR / "fiducials.json"
PRIORS_PATH = CONFIG_DIR / "priors.json"
NETWORKS_PATH = CONFIG_DIR / "networks.json"
ROOT_LAYERS = (FIDUCIALS_PATH, PRIORS_PATH, NETWORKS_PATH)

#: Presentation settings, read by `astrogwb.paper.plotting`. Deliberately *not*
#: a run-config layer: nothing a run samples depends on it.
PLOTTING_PATH = CONFIG_DIR / "plotting.json"

#: Catalog layer 0: the ``[waveform]`` block every catalog inherits. Named
#: rather than globbed because it sits next to the run JSON files and
#: ``config/plotting.json``, which must not enter a catalog merge. Deliberately
#: *not* a run-config layer: ``RunConfig`` is ``extra="forbid"``.
WAVEFORM_PATH = CONFIG_DIR / "waveform.json"

ANALYSIS_DIR = Path("config/analysis")
BASE_DIR = ANALYSIS_DIR / "base"
RUNS_DIR = ANALYSIS_DIR / "runs"
EXPERIMENT_BASE = "_base.toml"

#: Catalogs follow the same two-layer shape as runs: a shared base and one
#: file per named catalog, whose stem is the name.
CATALOGS_DIR = Path("config/catalogs")
CATALOG_BASE_DIR = CATALOGS_DIR / "base"
CATALOG_DEFS_DIR = CATALOGS_DIR / "defs"

#: Root of everything the workflow writes, so the three output roots below are
#: composed rather than re-typed. Relative to the working directory, as every
#: path in this module is.
BASE_OUT_DIR = Path("outputs")

CHAINS_ROOT = BASE_OUT_DIR / "chains"
CATALOGS_ROOT = BASE_OUT_DIR / "catalogs"
FIGURES_DIR = BASE_OUT_DIR / "figures"


def _merge_run_overlay(
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
    :func:`_merge_run_overlay`, whose prior-replacement rule is domain-specific.
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
        merged = _merge_run_overlay(merged, load_mapping(path))
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


def root_config_paths(root: Path | None = None) -> tuple[Path, ...]:
    """Layer 0: the top-level ``config/*.json`` files, in merge order.

    Named explicitly rather than globbed. These three are a fixed contract --
    `astrogwb.paper.config` exposes each one through an accessor -- and
    ``config/plotting.json`` sits in the same directory without being a run
    layer, which a glob would sweep in.
    """
    resolved = root or Path()
    paths = tuple(resolved / path for path in ROOT_LAYERS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("missing shared config layer(s): " + ", ".join(missing))
    return paths


def base_config_paths(root: Path | None = None) -> tuple[Path, ...]:
    """Every shared layer a run inherits, in merge order.

    Layer 0 (``config/*.json``) first, then ``config/analysis/base/*.toml``.
    The TOML half is sorted for determinism only: the base files partition
    disjoint top-level keys, so the order does not change the outcome. The JSON
    half is ordered by :data:`ROOT_LAYERS` and partitions disjoint keys too --
    ``fiducials``, ``priors``, ``networks`` -- so the whole prefix is
    order-insensitive in practice and ordered anyway for reproducibility.
    """
    directory = (root or Path()) / BASE_DIR
    paths = tuple(sorted(directory.glob("*.toml")))
    if not paths:
        raise ValueError(f"{directory} declares no base config files")
    return (*root_config_paths(root), *paths)


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


CATALOG_ROLES = ("injection", "proposal")


def catalog_base_paths(root: Path | None = None) -> tuple[Path, ...]:
    """Every shared catalog layer, in merge order.

    Layer 0 is ``config/waveform.json`` -- named, because a glob of
    ``config/*.json`` would also sweep in the run tables and
    ``config/plotting.json``. Then ``config/catalogs/base/*.toml``. Only
    ``config/catalogs/defs/md-taylorf2-s41-n32768.toml`` overrides anything in
    the waveform block (the approximant); the rest of the tree inherits it
    verbatim. The stored band matches ``config/analysis/base/model.toml``'s
    ``[analysis]`` ``f_min`` / ``f_max``: the catalog grid *is* the array
    every model is evaluated on. ``sampling_frequency`` is the waveform
    backend's Nyquist, not the stored grid.
    """
    resolved = root or Path()
    waveform = resolved / WAVEFORM_PATH
    if not waveform.is_file():
        raise ValueError(f"missing shared catalog layer: {waveform}")
    directory = resolved / CATALOG_BASE_DIR
    paths = tuple(sorted(directory.glob("*.toml")))
    if not paths:
        raise ValueError(f"{directory} declares no base config files")
    return (waveform, *paths)


def catalog_config_paths(name: str, *, root: Path | None = None) -> tuple[Path, ...]:
    """The ordered layer files that make up one catalog's config.

    Mirrors :func:`run_config_paths`: the ``Snakefile`` declares exactly these
    as the catalog rule's inputs, so editing ``config/waveform.json``
    invalidates every catalog.
    """
    resolved = root or Path()
    definition = resolved / CATALOG_DEFS_DIR / f"{name}.toml"
    if not definition.is_file():
        raise ValueError(f"unknown catalog {name}: {definition} does not exist")
    return (*catalog_base_paths(resolved), definition)


def discover_catalog_names(root: Path | None = None) -> tuple[str, ...]:
    """Every declared catalog name, sorted. Stems of ``config/catalogs/defs``."""
    directory = (root or Path()) / CATALOG_DEFS_DIR
    names = tuple(sorted(path.stem for path in directory.glob("*.toml")))
    if not names:
        raise ValueError(f"{directory} declares no catalog configs")
    return names


def _catalog_names(raw: Mapping[str, Any]) -> dict[str, str]:
    """The catalog each role names, from a raw config's ``[catalog]`` block."""
    catalog = raw.get("catalog")
    if not isinstance(catalog, Mapping):
        raise TypeError("run config must define a [catalog] table")
    names: dict[str, str] = {}
    for role in CATALOG_ROLES:
        name = catalog.get(role)
        if not isinstance(name, str) or not name:
            raise TypeError(f"catalog.{role} must be a catalog name")
        names[role] = name
    return names


def resolve_catalog_names(
    experiment: str, run: str, *, root: Path | None = None
) -> dict[str, str]:
    """The injection and proposal catalogs a run names, keyed by role.

    The ``Snakefile`` calls this to declare ``run_mcmc``'s catalog inputs, which
    is why it works off the merged mapping rather than a validated
    :class:`~astrogwb.paper.config.mcmc.RunConfig`: the DAG must be buildable
    without paying for full validation of all 26 runs.
    """
    return _catalog_names(assemble_run(experiment, run, root=root))


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
        # Resolved through the run's own merge rather than by looking the label
        # up in `config/networks.json` directly. The two agree today only
        # because every network run happens to be named after the network it
        # uses, which is a property of the tree and not a derivation: a direct
        # lookup would report SNRs for one network beside a chain sampled on
        # another the moment a run changed its `network`.
        merged = assemble_run(experiment, name, root=root)
        analysis = merged.get("analysis") or {}
        network_name = analysis.get("network")
        if not network_name:
            raise ValueError(f"{experiment}/{name} declares no analysis.network")
        table = merged.get("networks") or {}
        detectors = table.get(network_name)
        if not detectors:
            raise ValueError(
                f"{experiment}/{name} names network {network_name!r}, which "
                f"config/networks.json does not declare; known networks: "
                f"{sorted(table)}"
            )
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
    ``scripts/run_mcmc.py``.
    """
    paths: list[Path] = list(args.config)
    logger.info(
        "Config layers (merge order): %s", " -> ".join(str(path) for path in paths)
    )
    return merge_config_layers(paths)
