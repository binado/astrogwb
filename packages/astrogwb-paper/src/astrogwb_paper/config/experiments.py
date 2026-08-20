"""Committed inventory of reproducibility experiments and their MCMC runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrogwb_paper.config.catalogs import (
    catalog_recipe,
    load_catalogs,
    proposal_config,
)
from astrogwb_paper.config.loading import load_inventory, merge_run_overlay
from astrogwb_paper.paths import paper_project_root

DEFAULT_CATALOG = "bns-n16384-eps=0-df1"
EXPERIMENTS_PATH = Path("inputs/experiments.yaml")
_REQUIRED_SECTIONS = ("base", "experiments")
# `networks` exists purely to anchor detector lists for the runs to alias.
_OPTIONAL_SECTIONS = ("networks",)


@dataclass(frozen=True)
class Experiment:
    """One explicit experiment loaded from the shared YAML inventory."""

    name: str
    path: Path
    runs: tuple[str, ...]
    run_overlays: dict[str, dict[str, Any]]
    defaults: dict[str, Any]

    @property
    def run_target(self) -> str:
        """Return the Snakemake target that runs this experiment's MCMCs."""
        return f"run_experiment_{self.name.replace('-', '_')}"

    def catalog_for(self, run: str) -> str:
        """Return the name of the prebuilt catalog consumed by ``run``."""
        catalog = self.run_overlays[self._require_run(run)].get("catalog")
        return str(catalog) if catalog else DEFAULT_CATALOG

    def merged_config_path(self, run: str) -> Path:
        """Return the generated canonical JSON config path for ``run``."""
        return Path("outputs/configs") / self.name / f"{self._require_run(run)}.json"

    def chain_path(self, run: str) -> Path:
        """Return the generated NetCDF chain path for ``run``."""
        return Path("outputs/chains") / self.name / f"{self._require_run(run)}.nc"

    def chain_paths(self) -> list[str]:
        """Return every chain path owned by this experiment."""
        return [str(self.chain_path(run)) for run in self.runs]

    def _require_run(self, run: str) -> str:
        if run not in self.runs:
            raise ValueError(f"unknown run {self.name}/{run}")
        return run


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


def _load_inventory(path: Path) -> dict[str, Any]:
    return load_inventory(
        path, required=_REQUIRED_SECTIONS, optional=_OPTIONAL_SECTIONS
    )


def load_base(path: Path | None = None) -> dict[str, Any]:
    """Load the shared run configuration from the YAML inventory."""
    return dict(_load_inventory(path or inventory_path())["base"])


def load_experiments(path: Path | None = None) -> dict[str, Experiment]:
    """Load every committed experiment from the YAML inventory."""
    resolved = path or inventory_path()
    raw = _load_inventory(resolved)
    experiments_raw = raw["experiments"]
    assert isinstance(experiments_raw, Mapping)
    experiments: dict[str, Experiment] = {}
    for name, specification in experiments_raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{resolved} has an invalid experiment name")
        if not isinstance(specification, Mapping):
            raise TypeError(f"{resolved} experiment {name!r} must be a mapping")
        experiments[name] = load_experiment(name, specification, resolved)
    _check_catalogs_exist(experiments, resolved)
    return experiments


def _check_catalogs_exist(experiments: Mapping[str, Experiment], path: Path) -> None:
    """Reject a run naming a catalog the catalog inventory does not declare.

    Without this the typo only surfaces at workflow time, as a Snakemake
    missing-input error for a path no rule can produce.
    """
    known = load_catalogs()
    choices = ", ".join(known)
    for specification in experiments.values():
        for run in specification.runs:
            catalog = specification.catalog_for(run)
            if catalog not in known:
                raise ValueError(
                    f"{path} run {specification.name}/{run} names unknown "
                    f"catalog {catalog!r}; choose from {choices}"
                )


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
    assembled = merge_run_overlay(base, merged)
    proposal = proposal_config(catalog_recipe(spec.catalog_for(run)))
    _validate_proposal_matches_run(spec, run, assembled, proposal)
    assembled["proposal"] = proposal
    return assembled


def _validate_proposal_matches_run(
    spec: Experiment,
    run: str,
    assembled: Mapping[str, Any],
    proposal: Mapping[str, float | int],
) -> None:
    """Keep generation-time proposal constants aligned with run fiducials."""
    fiducials = assembled.get("fiducials")
    cosmology = assembled.get("cosmology")
    if not isinstance(fiducials, Mapping) or not isinstance(cosmology, Mapping):
        raise TypeError(f"{spec.name}/{run} must define fiducials and cosmology")
    mismatches = [
        name
        for name in ("H0", "Omega_m", "gamma", "kappa", "z_peak")
        if float(fiducials[name]) != float(proposal[name])
    ]
    if float(cosmology["z_min"]) != float(proposal["z_min"]) or float(
        cosmology["z_max"]
    ) != float(proposal["z_max"]):
        mismatches.append("redshift support")
    if mismatches:
        raise ValueError(
            f"{spec.name}/{run} proposal does not match run settings: "
            + ", ".join(mismatches)
        )
