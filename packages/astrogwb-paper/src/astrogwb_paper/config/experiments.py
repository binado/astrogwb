"""Committed inventory of reproducibility experiments and their MCMC runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrogwb_paper.config.loading import load_mapping, merge_run_overlay
from astrogwb_paper.paths import paper_project_root

DEFAULT_CATALOG = Path("outputs/catalogs/bns-n16384-df1.h5")
_EXPERIMENT_ONLY_KEYS = frozenset({"runs"})


@dataclass(frozen=True)
class Experiment:
    """One explicit experiment loaded from ``experiments/<name>.toml``."""

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


def load_experiment(path: Path) -> Experiment:
    """Load one experiment TOML into an :class:`Experiment`."""
    raw = load_mapping(path)
    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, Mapping) or not runs_raw:
        raise ValueError(f"{path} must define a non-empty [runs] table")
    run_overlays = {
        name: dict(overlay) if isinstance(overlay, Mapping) else {}
        for name, overlay in runs_raw.items()
    }
    return Experiment(
        name=path.stem,
        path=path,
        runs=tuple(run_overlays),
        run_overlays=run_overlays,
        defaults={
            key: value for key, value in raw.items() if key not in _EXPERIMENT_ONLY_KEYS
        },
    )


def load_experiments(root: str | None = None) -> dict[str, Experiment]:
    """Discover committed experiments from ``experiments/*.toml``."""
    directory = Path(root) if root is not None else paper_project_root() / "experiments"
    leftovers = sorted(path for path in directory.glob("*/*.toml"))
    if leftovers:
        nested = ", ".join(str(path.relative_to(directory)) for path in leftovers)
        raise ValueError(
            f"experiment inventory must be flat *.toml files; found nested {nested}"
        )
    return {
        spec.name: spec
        for spec in (load_experiment(path) for path in sorted(directory.glob("*.toml")))
    }


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


def config_path(experiment_name: str, run: str) -> Path:
    """Return the committed experiment TOML path."""
    spec = experiment(experiment_name)
    spec.catalog_for(run)
    return Path("experiments") / f"{spec.name}.toml"


def merged_config_path(experiment_name: str, run: str) -> Path:
    """Return the generated canonical JSON path."""
    experiment(experiment_name).catalog_for(run)
    return Path("outputs/configs") / experiment_name / f"{run}.json"


def chain_path(experiment_name: str, run: str) -> Path:
    """Return the generated NetCDF chain path."""
    experiment(experiment_name).catalog_for(run)
    return Path("outputs/chains") / experiment_name / f"{run}.nc"


def sidecar_path(experiment_name: str, run: str) -> Path:
    """Return the generated provenance sidecar path."""
    return chain_path(experiment_name, run).with_suffix(".json")


def chain_paths(experiment_name: str) -> list[str]:
    """Return every chain path owned by an experiment."""
    spec = experiment(experiment_name)
    return [str(chain_path(spec.name, run)) for run in spec.runs]
