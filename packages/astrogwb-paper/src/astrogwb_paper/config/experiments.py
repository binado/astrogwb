"""Committed inventory of reproducibility experiments and their MCMC runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrogwb_paper.config.loading import load_mapping, merge_run_overlay
from astrogwb_paper.paths import paper_project_root

DEFAULT_CATALOG = Path("outputs/catalogs/bns-n16384-df1.h5")
EXPERIMENTS_PATH = Path("inputs/experiments.yaml")
_INVENTORY_KEYS = frozenset({"base", "experiments"})


@dataclass(frozen=True)
class Experiment:
    """One explicit experiment loaded from the shared YAML inventory."""

    name: str
    path: Path
    runs: tuple[str, ...]
    run_overlays: dict[str, dict[str, Any]]
    defaults: dict[str, Any]

    @property
    def target(self) -> str:
        """Return the complete Snakemake target name."""
        return self.name.replace("-", "_")

    @property
    def chains_target(self) -> str:
        """Return the chains-only Snakemake target name."""
        return f"{self.target}_chains"

    def catalog_for(self, run: str) -> Path:
        """Return the prebuilt catalog consumed by ``run``."""
        if run not in self.runs:
            raise ValueError(f"unknown run {self.name}/{run}")
        catalog = self.run_overlays[run].get("catalog")
        return Path(catalog) if catalog else DEFAULT_CATALOG


def load_experiment(name: str, raw: Mapping[str, Any], path: Path) -> Experiment:
    """Load one named experiment mapping into an :class:`Experiment`."""
    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, Mapping) or not runs_raw:
        raise ValueError(f"{path} experiment {name!r} must define non-empty runs")
    run_overlays: dict[str, dict[str, Any]] = {}
    for run, overlay in runs_raw.items():
        if not isinstance(run, str) or not run:
            raise ValueError(f"{path} experiment {name!r} has an invalid run name")
        if overlay is not None and not isinstance(overlay, Mapping):
            raise ValueError(f"{path} run {name}/{run} must be a mapping")
        run_overlays[run] = dict(overlay or {})
    return Experiment(
        name=name,
        path=path,
        runs=tuple(run_overlays),
        run_overlays=run_overlays,
        defaults={key: value for key, value in raw.items() if key != "runs"},
    )


def inventory_path(root: Path | None = None) -> Path:
    """Return the committed MCMC inventory path."""
    return (root or paper_project_root()) / EXPERIMENTS_PATH


def load_inventory(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the top-level YAML inventory."""
    resolved = path or inventory_path()
    raw = load_mapping(resolved)
    unknown = sorted(set(raw) - _INVENTORY_KEYS)
    if unknown:
        raise ValueError(f"{resolved} has unknown top-level keys: {', '.join(unknown)}")
    base = raw.get("base")
    experiments = raw.get("experiments")
    if not isinstance(base, Mapping) or not base:
        raise ValueError(f"{resolved} must define a non-empty base mapping")
    if not isinstance(experiments, Mapping) or not experiments:
        raise ValueError(f"{resolved} must define non-empty experiments")
    return raw


def load_base(path: Path | None = None) -> dict[str, Any]:
    """Load the shared run configuration from the YAML inventory."""
    return dict(load_inventory(path)["base"])


def load_experiments(path: Path | None = None) -> dict[str, Experiment]:
    """Load every committed experiment from the YAML inventory."""
    resolved = path or inventory_path()
    raw = load_inventory(resolved)
    experiments_raw = raw["experiments"]
    assert isinstance(experiments_raw, Mapping)
    experiments: dict[str, Experiment] = {}
    for name, specification in experiments_raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{resolved} has an invalid experiment name")
        if not isinstance(specification, Mapping):
            raise TypeError(f"{resolved} experiment {name!r} must be a mapping")
        experiments[name] = load_experiment(name, specification, resolved)
    return experiments


def experiment(name: str) -> Experiment:
    """Return a named experiment or raise a user-facing error."""
    experiments = load_experiments()
    try:
        return experiments[name]
    except KeyError:
        choices = ", ".join(experiments)
        raise ValueError(
            f"unknown experiment {name!r}; choose from {choices}"
        ) from None


def overlay_for(
    spec: Experiment,
    run: str,
    *,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge base, experiment defaults, and one run overlay into a raw config."""
    if run not in spec.runs:
        raise ValueError(f"unknown run {spec.name}/{run}")
    run_overlay = {
        key: value for key, value in spec.run_overlays[run].items() if key != "catalog"
    }
    merged = merge_run_overlay(spec.defaults, run_overlay)
    if base is None:
        return merged
    return merge_run_overlay(base, merged)


def merged_config_path(experiment_name: str, run: str) -> Path:
    """Return the generated canonical JSON path."""
    experiment(experiment_name).catalog_for(run)
    return Path("outputs/configs") / experiment_name / f"{run}.json"


def chain_path(experiment_name: str, run: str) -> Path:
    """Return the generated NetCDF chain path."""
    experiment(experiment_name).catalog_for(run)
    return Path("outputs/chains") / experiment_name / f"{run}.nc"


def chain_paths(experiment_name: str) -> list[str]:
    """Return every chain path owned by an experiment."""
    spec = experiment(experiment_name)
    return [str(chain_path(spec.name, run)) for run in spec.runs]
