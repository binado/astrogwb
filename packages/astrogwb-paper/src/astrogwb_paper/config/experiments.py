"""Committed inventory of reproducibility experiments and their MCMC runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CATALOG = Path("outputs/catalogs/bns-n16384-df1.h5")

DETECTOR_NETWORKS: dict[str, tuple[str, ...]] = {
    "ET-triangular": ("E1", "E2", "E3"),
    "ET-triangular-CE-Hanford": ("E1", "E2", "E3", "C1"),
    "ET-2L-aligned": ("S1", "R1"),
    "ET-2L-aligned-CE-Hanford": ("S1", "R1", "C1"),
    "ET-2L-misaligned": ("S2", "R2"),
    "ET-2L-misaligned-CE-Hanford": ("S2", "R2", "C1"),
}


@dataclass(frozen=True)
class Experiment:
    """One explicit experiment and the runs it owns."""

    name: str
    runs: tuple[str, ...]
    has_figures: bool = False
    catalogs: tuple[tuple[str, Path], ...] = ()

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
        return dict(self.catalogs).get(run, DEFAULT_CATALOG)


EXPERIMENTS: dict[str, Experiment] = {
    experiment.name: experiment
    for experiment in (
        Experiment(
            name="H0-all-detectors",
            runs=tuple(DETECTOR_NETWORKS),
            has_figures=True,
        ),
        Experiment(
            name="modified-propagation-all-detectors",
            runs=(*DETECTOR_NETWORKS, "Xi_0", "Xi_0-H0"),
            has_figures=True,
        ),
        Experiment(
            name="H0-merger-rate",
            runs=("fixed", "sampled"),
            has_figures=True,
        ),
        Experiment(
            name="H0-omega-m",
            runs=("H0-Omega_m",),
            has_figures=True,
        ),
        Experiment(
            name="astrophysical-parameters",
            runs=("Madau-Dickinson",),
        ),
        Experiment(
            name="star-formation-peak",
            runs=("z_peak",),
        ),
        Experiment(
            name="variable-injection-size",
            runs=("n8192", "n16384", "n32768"),
            catalogs=(
                ("n8192", Path("outputs/catalogs/bns-n8192-df1.h5")),
                ("n16384", Path("outputs/catalogs/bns-n16384-df1.h5")),
                ("n32768", Path("outputs/catalogs/bns-n32768-df1.h5")),
            ),
        ),
    )
}


def experiment(name: str) -> Experiment:
    """Return a named experiment or raise a user-facing error."""
    try:
        return EXPERIMENTS[name]
    except KeyError:
        choices = ", ".join(EXPERIMENTS)
        raise ValueError(
            f"unknown experiment {name!r}; choose from {choices}"
        ) from None


def config_path(experiment_name: str, run: str) -> Path:
    """Return the committed one-run TOML path."""
    experiment(experiment_name).catalog_for(run)
    return Path("experiments") / experiment_name / f"mcmc.{run}.toml"


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
