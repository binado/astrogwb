"""Discover, locate, and merge MCMC run configs from the ``config/`` tree.

Filenames are the mapping. ``config/runs/<experiment>/<run>.json`` samples into
``outputs/chains/<experiment>/<run>.nc``; no inventory file translates between
the two. That convention is what let the previous ``inputs/experiments.yaml``
registry -- and the ten ``Snakefile`` helpers that read it -- go away.

A run config is three layers merged in order:

0. ``config/{analysis,fiducials,networks,priors,sampler,waveform,population}.json``
   -- the shared values, one file per top-level block of a run config, each a single-key
   object whose key is its own stem. :func:`base_config_paths` is that list, so
   a caller that wants "everything shared" asks for it once.
1. ``config/runs/<experiment>/_base.json`` -- the experiment override.
2. ``config/runs/<experiment>/<run>.json`` -- the run override.

Every layer is JSON. :func:`merge_config_layers` folds them with ``knf``, the
engine behind the ``knf`` CLI, so the shell and Python spell one merge rule:
``knf <layers> --shallow 'priors.*'`` prints what a run resolves to.

A run also owns its catalogs. ``[analysis.catalog]`` holds a partial spec per
role, and :func:`resolve_catalog_blocks` completes it from the run's own
``[waveform]``, ``[population]`` and ``[fiducials]``; the result is keyed by
:class:`~astrogwb.metadata.CatalogMetadata` and names the catalog file.

:data:`EXPERIMENT_BASE` is required in every experiment directory rather than
optional: a conditional Snakemake input complicates the DAG for no gain.

There is no assembled-config artifact. Every *run* entrypoint, ``run_mcmc``
included, is handed its layer files on argv (see :func:`add_config_arguments`)
and merges them in process; :func:`assemble_run` is the convenience wrapper for the notebooks and
for the validation gate, which address a run by name rather than by path.
``scripts/generate_catalog.py`` is handed a resolved request as JSON instead.

**stdlib plus knf, and deliberately so.** Resolution here is plain dict
merging, so it has one implementation that the ``Snakefile``, ``RunConfig`` and
the notebooks all share.
Validating and keying the result -- which reaches pydantic -- lives in
:mod:`astrogwb.paper.config.catalogs`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import knf

from astrogwb.paper.utils import deep_merge, require_json

if TYPE_CHECKING:
    from astrogwb.paper.plotting import Network

logger = logging.getLogger(__name__)

#: Relative to the working directory, which for the workflow and every script
#: is the repository root. Library code names no absolute path and does not go
#: looking for a checkout: the caller's cwd is the answer.
CONFIG_DIR = Path("config")

#: Every shared run layer: one file per top-level block of a run config, each
#: a single-key object whose key is its own stem. ``fiducials``, ``priors`` and
#: ``networks`` are also read directly by the notebooks and figure scripts
#: through `astrogwb.paper.config`. All five are ordinary merge layers, so
#: nothing here special-cases them.
ANALYSIS_PATH = CONFIG_DIR / "analysis.json"
FIDUCIALS_PATH = CONFIG_DIR / "fiducials.json"
NETWORKS_PATH = CONFIG_DIR / "networks.json"
PRIORS_PATH = CONFIG_DIR / "priors.json"
SAMPLER_PATH = CONFIG_DIR / "sampler.json"
#: The ``[waveform]`` and ``[population]`` blocks every catalog a run draws
#: inherits. ``population`` here is the *catalog* default, not the analysis
#: target, which is ``analysis.population``.
WAVEFORM_PATH = CONFIG_DIR / "waveform.json"
POPULATION_PATH = CONFIG_DIR / "population.json"
ROOT_LAYERS = (
    ANALYSIS_PATH,
    FIDUCIALS_PATH,
    NETWORKS_PATH,
    PRIORS_PATH,
    SAMPLER_PATH,
    WAVEFORM_PATH,
    POPULATION_PATH,
)

#: Presentation settings, read by `astrogwb.paper.plotting`. Deliberately *not*
#: a run-config layer: nothing a run samples depends on it.
PLOTTING_PATH = CONFIG_DIR / "plotting.json"

RUNS_DIR = CONFIG_DIR / "runs"
EXPERIMENT_BASE = "_base.json"

#: Root of everything the workflow writes, so the three output roots below are
#: composed rather than re-typed. Relative to the working directory, as every
#: path in this module is.
BASE_OUT_DIR = Path("outputs")

CHAINS_ROOT = BASE_OUT_DIR / "chains"
CATALOGS_ROOT = BASE_OUT_DIR / "catalogs"
SPECTRA_ROOT = BASE_OUT_DIR / "spectra"
FIGURES_DIR = BASE_OUT_DIR / "figures"


#: The one exception to a deep merge, as a ``knf`` key-path glob: each
#: ``[priors.<param>]`` table replaces the inherited one wholesale. A run that
#: swaps a uniform prior for a normal one must not key-merge onto it and keep
#: the uniform's ``low`` / ``high`` beside ``loc`` / ``scale``.
#: ``tests/paper/test_runs.py`` shows the deep merge failing on the one run
#: that overrides a prior.
PRIOR_SHALLOW = "priors.*"


def merge_config_layers(paths: Sequence[Path]) -> dict[str, Any]:
    """Fold run-config layer files into one raw mapping, in the order given.

    A deep merge, left to right -- arrays and scalars replace -- except at
    :data:`PRIOR_SHALLOW`. This is a *run-config* parser, not generic config
    infrastructure: the prior rule is domain-specific. The shared layers
    declare disjoint top-level blocks, so one fold serves every layer.

    Order is the caller's responsibility and it is not recoverable from the
    result, so :func:`load_merged_config` logs it.
    """
    if not paths:
        raise ValueError("no config layers given")
    for path in paths:
        require_json(path)
    return knf.load(list(paths), shallow=PRIOR_SHALLOW)


def discover_runs(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Return ``{experiment: (run, ...)}`` by globbing the runs tree.

    :data:`EXPERIMENT_BASE` is the experiment override, not a run, so it is
    excluded.
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
                for path in directory.glob("*.json")
                if path.name != EXPERIMENT_BASE
            )
        )
        if not names:
            raise ValueError(f"{directory} declares no runs")
        runs[experiment] = names
    return runs


def base_config_paths(root: Path | None = None) -> tuple[Path, ...]:
    """Every shared layer a run inherits, in merge order: :data:`ROOT_LAYERS`.

    Named explicitly rather than globbed. ``config/plotting.json`` sits in the
    same directory without being a run layer, which a glob of
    ``config/*.json`` would sweep in -- and ``RunConfig`` is
    ``extra="forbid"``, so it would sweep it in loudly. The seven declare
    disjoint top-level blocks, so the order among them does not change the
    outcome; it is fixed anyway for reproducibility.
    """
    resolved = root or Path()
    paths = tuple(resolved / path for path in ROOT_LAYERS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("missing shared config layer(s): " + ", ".join(missing))
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
    run_path = directory / f"{run}.json"
    if not run_path.is_file():
        raise ValueError(f"unknown run {experiment}/{run}: {run_path} does not exist")
    experiment_base = directory / EXPERIMENT_BASE
    if not experiment_base.is_file():
        raise ValueError(f"{directory} is missing a required {EXPERIMENT_BASE}")
    return (*base_config_paths(resolved), experiment_base, run_path)


def load_base(root: Path | None = None) -> dict[str, Any]:
    """Merge every shared run layer into one mapping."""
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

#: The blocks a role's spec may override, each deep-merged over the run's own
#: block of the same name. ``seed`` and ``num_samples`` have no default: they
#: belong to a particular draw, so every role states them.
CATALOG_OVERRIDABLE_BLOCKS = ("waveform", "population", "fiducials")


def resolve_catalog_blocks(raw: Mapping[str, Any], role: str) -> dict[str, Any]:
    """The complete catalog blocks one role of a merged run config asks for.

    A role's spec in ``[analysis.catalog]`` is partial: it states the draw's
    ``seed`` and ``num_samples`` and overrides whatever else differs. The rest
    is the run's own ``[waveform]``, ``[population]`` and ``[fiducials]``, so
    the injection is drawn at the very hyperparameters the run initializes at.
    The overrides are recursive merges, so a guarded proposal that names another population still inherits
    the shared redshift window.

    Returns plain blocks, not a validated request: this module is stdlib-only,
    and :meth:`astrogwb.metadata.CatalogMetadata.from_blocks` is where they are
    checked and keyed.
    """
    if role not in CATALOG_ROLES:
        raise ValueError(f"unknown catalog role {role!r}; roles are {CATALOG_ROLES}")
    analysis = raw.get("analysis")
    catalog = analysis.get("catalog") if isinstance(analysis, Mapping) else None
    spec = catalog.get(role) if isinstance(catalog, Mapping) else None
    if not isinstance(spec, Mapping):
        raise TypeError(f"run config must define an [analysis.catalog.{role}] table")
    blocks: dict[str, Any] = {}
    for block in CATALOG_OVERRIDABLE_BLOCKS:
        inherited = raw.get(block)
        if not isinstance(inherited, Mapping):
            raise TypeError(f"run config must define a [{block}] block")
        blocks[block] = deep_merge(inherited, spec.get(block) or {})
    for name in ("seed", "num_samples"):
        if name not in spec:
            raise TypeError(f"analysis.catalog.{role} must declare {name}")
        blocks[name] = spec[name]
    return blocks


def catalog_blocks(
    experiment: str, run: str, role: str, *, root: Path | None = None
) -> dict[str, Any]:
    """:func:`resolve_catalog_blocks` for a run addressed by name."""
    return resolve_catalog_blocks(assemble_run(experiment, run, root=root), role)


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
            "One run-config layer file, in merge order; repeat once per "
            "layer (the shared config/*.json, then the experiment _base.json, "
            "then the run)."
        ),
    )


def load_merged_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the ``--config`` layers an entrypoint was handed, order preserved.

    Merge order is the caller's to get right now that no single function owns
    it, and a wrong-but-valid order fails silently, so the resolved order is
    logged before the merge. The workflow passes :func:`run_config_paths`, which
    is the order by construction.
    """
    paths: list[Path] = list(args.config)
    logger.info(
        "Config layers (merge order): %s", " -> ".join(str(path) for path in paths)
    )
    return merge_config_layers(paths)
