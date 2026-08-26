"""Discover and assemble MCMC runs from the ``config/analysis/`` tree.

Filenames are the mapping. ``config/analysis/runs/<experiment>/<run>.toml``
assembles into ``outputs/configs/<experiment>/<run>.json`` and samples into
``outputs/chains/<experiment>/<run>.nc``; no inventory file translates between
the two. That convention is what let the previous ``inputs/experiments.yaml``
registry -- and the ten ``Snakefile`` helpers that read it -- go away.

A run config is three layers merged in order:

1. ``config/analysis/base/*.toml`` -- settings every run shares.
2. ``config/analysis/runs/<experiment>/_base.toml`` -- the experiment override.
3. ``config/analysis/runs/<experiment>/<run>.toml`` -- the run override.

``_base.toml`` is required in every experiment directory rather than optional:
a conditional Snakemake input complicates the DAG for no gain.

stdlib + pydantic only, and the ``Snakefile`` imports :func:`assemble_run`
directly to resolve each run's bank inputs -- so an error raised here breaks DAG
construction for *every* target, not just the offending run. Keep it
dependency-light.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from astrogwb_paper.config.banks import BankGenerationConfig, discover_banks
from astrogwb_paper.config.mcmc import RunConfig
from astrogwb_paper.paths import paper_project_root
from astrogwb_paper.utils import deep_merge, load_mapping

ANALYSIS_DIR = Path("config/analysis")
BASE_DIR = ANALYSIS_DIR / "base"
RUNS_DIR = ANALYSIS_DIR / "runs"
EXPERIMENT_BASE = "_base.toml"

CONFIGS_ROOT = Path("outputs/configs")
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


def discover_runs(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Return ``{experiment: (run, ...)}`` by globbing the runs tree.

    ``_base.toml`` is the experiment override, not a run, so it is excluded.
    """
    resolved = root or paper_project_root()
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


def load_base(root: Path | None = None) -> dict[str, Any]:
    """Merge every ``config/analysis/base/*.toml`` into one mapping.

    The base files partition disjoint top-level keys, so the sorted-glob order
    only matters for determinism, not for outcome.
    """
    directory = (root or paper_project_root()) / BASE_DIR
    paths = sorted(directory.glob("*.toml"))
    if not paths:
        raise ValueError(f"{directory} declares no base config files")
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_mapping(path))
    return merged


def assemble_run(
    experiment: str, run: str, *, root: Path | None = None
) -> dict[str, Any]:
    """Merge base, experiment, and run layers into one raw run config.

    Uses :func:`merge_run_overlay` rather than a plain deep merge: each
    ``[priors.<param>]`` table replaces the layer below it wholesale. That is
    load-bearing, not incidental -- key-merging a normal prior onto a uniform
    one would leave stale ``low`` / ``high`` behind.
    """
    resolved = root or paper_project_root()
    directory = resolved / RUNS_DIR / experiment
    run_path = directory / f"{run}.toml"
    if not run_path.is_file():
        raise ValueError(f"unknown run {experiment}/{run}: {run_path} does not exist")
    experiment_base = directory / EXPERIMENT_BASE
    if not experiment_base.is_file():
        raise ValueError(f"{directory} is missing a required {EXPERIMENT_BASE}")

    merged = merge_run_overlay(load_base(resolved), load_mapping(experiment_base))
    return merge_run_overlay(merged, load_mapping(run_path))


def run_bank_names(experiment: str, run: str, *, root: Path | None = None) -> list[str]:
    """Every distinct bank a run's injection and proposal catalogs draw from.

    The ``Snakefile`` calls this to declare ``run_mcmc``'s bank inputs, which is
    why it works off the merged mapping rather than a validated
    :class:`~astrogwb_paper.config.mcmc.RunConfig`: the DAG must be buildable
    without paying for full validation of all 26 runs.
    """
    return catalog_bank_names(assemble_run(experiment, run, root=root))


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


def check_bank_references(
    config: RunConfig,
    *,
    label: str,
    banks: Mapping[str, BankGenerationConfig] | None = None,
) -> None:
    """Reject a run naming an unknown bank, or a self-correlated mixture.

    Runs at assemble time, against the committed bank configs rather than the
    built bank files, so a typo fails without building anything expensive.
    Without it the typo would only surface as a Snakemake wildcard that matches
    no rule.
    """
    known = banks if banks is not None else discover_banks()
    for role, spec in (
        ("injection", config.catalog.injection),
        ("proposal", config.catalog.proposal),
    ):
        md = _require_bank(spec.md_bank, known, label=f"{label} catalog.{role}")
        if spec.uniform_bank is None:
            continue
        uniform = _require_bank(
            spec.uniform_bank, known, label=f"{label} catalog.{role}"
        )
        assert spec.mixture_seed is not None  # enforced by CatalogSpec
        seeds = (md.seed, uniform.seed, spec.mixture_seed)
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                f"{label} catalog.{role}: md_bank seed, uniform_bank seed, and "
                f"mixture_seed must all be distinct (got {seeds})"
            )


def _require_bank(
    name: str, banks: Mapping[str, BankGenerationConfig], *, label: str
) -> BankGenerationConfig:
    try:
        return banks[name]
    except KeyError:
        choices = ", ".join(banks)
        raise ValueError(
            f"{label} names unknown bank {name!r}; choose from {choices}"
        ) from None


def config_path(experiment: str, run: str) -> Path:
    """Return the assembled JSON config path for one run."""
    return CONFIGS_ROOT / experiment / f"{run}.json"


def chain_path(experiment: str, run: str) -> Path:
    """Return the NetCDF chain path for one run."""
    return CHAINS_ROOT / experiment / f"{run}.nc"


def run_target(experiment: str) -> str:
    """Return the Snakemake target that samples every run in an experiment."""
    return f"run_experiment_{experiment.replace('-', '_')}"
