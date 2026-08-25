"""Scientific inputs the paper figure scripts read for themselves (JAX-free).

Presentation -- which runs a figure shows, in what order, under which LaTeX
label -- is hard-coded in the figure scripts and in
:mod:`astrogwb_paper.plotting`. It is not configuration: changing a legend
label is a code change, reviewed with the plot it labels.

What stays here is the part with scientific consequences, and it is read from
the *assembled* run configs under ``outputs/configs/`` rather than from a
source inventory. A figure therefore reports exactly what its chains were
sampled with: the detector list attached to a network is the one that run's
config carried, and the fiducials and analysis grid come from a run rather than
from a shared ``base`` mapping that nothing guaranteed the run inherited.

Like :mod:`astrogwb_paper.config.loading`, this module imports neither JAX nor
``astrogwb``, so a figure script can resolve and validate its inputs before
touching a runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrogwb_paper.config.analysis import AnalysisGrid
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import CatalogSpec
from astrogwb_paper.config.runs import CONFIGS_ROOT, config_path
from astrogwb_paper.paths import paper_project_root

#: The run every figure reads shared settings from when it is not about one
#: specific run. Fiducials and the analysis grid come from ``base/`` for all 26
#: runs, so any run reports the same values -- but reading them from an
#: assembled config keeps figures honest about what was actually sampled.
REFERENCE_RUN = ("cosmological-parameters", "ET-2L-aligned-CE-Hanford")


@dataclass(frozen=True)
class Network:
    """One detector network: its run name, LaTeX label, and detector list."""

    name: str
    label: str
    detectors: tuple[str, ...]


def load_run_config(experiment: str, run: str) -> dict[str, Any]:
    """Read one assembled run config, with a pointer to what builds it."""
    path = paper_project_root() / config_path(experiment, run)
    if not path.is_file():
        raise FileNotFoundError(
            f"assembled config not found: {path}. Run "
            f"`snakemake {CONFIGS_ROOT}/{experiment}/{run}.json` (or "
            "`astrogwb-assemble-config --all`) first."
        )
    return load_mapping(path)


def resolve_networks(
    experiment_name: str,
    networks: Sequence[tuple[str, str]],
) -> tuple[Network, ...]:
    """Attach each ``(run, label)`` pair's detectors from its assembled config.

    Declaration order is preserved: it drives chain order, legend order, and
    the color/linestyle assignment in the detector-comparison figures.
    """
    if not networks:
        raise ValueError(f"{experiment_name} figure declares no detector networks")
    run_names = [name for name, _ in networks]
    duplicates = sorted({run for run in run_names if run_names.count(run) > 1})
    if duplicates:
        raise ValueError("duplicate detector network(s): " + ", ".join(duplicates))

    resolved: list[Network] = []
    for name, label in networks:
        analysis = load_run_config(experiment_name, name).get("analysis") or {}
        detectors = analysis.get("detectors")
        if not detectors:
            raise ValueError(f"{experiment_name}/{name} declares no analysis.detectors")
        resolved.append(Network(name, label, tuple(detectors)))
    return tuple(resolved)


def load_fiducials(run: tuple[str, str] = REFERENCE_RUN) -> dict[str, float]:
    """Load the ``fiducials`` mapping an assembled run was sampled at."""
    fiducials = load_run_config(*run).get("fiducials")
    if not fiducials:
        raise ValueError(f"{_label(run)} must define a non-empty [fiducials] table")
    return {str(name): float(value) for name, value in fiducials.items()}


def load_analysis_grid(run: tuple[str, str] = REFERENCE_RUN) -> AnalysisGrid:
    """Load the frequency band and redshift grid an assembled run was built on."""
    raw = load_run_config(*run)
    analysis = raw.get("analysis") or {}
    cosmology = raw.get("cosmology") or {}
    try:
        return AnalysisGrid(
            observation_time=float(raw["observation_time"]),
            f_min=float(analysis["f_min"]),
            f_max=float(analysis["f_max"]),
            minimum_redshift=float(cosmology["minimum_redshift"]),
            maximum_redshift=float(cosmology["maximum_redshift"]),
            n_grid=int(cosmology["n_grid"]),
        )
    except KeyError as error:
        raise ValueError(f"{_label(run)} is missing analysis setting {error}") from None


def load_injection_spec(run: tuple[str, str] = REFERENCE_RUN) -> CatalogSpec:
    """The injection catalog an assembled run composes.

    Every run shares one injection catalog -- it is the "observed" data -- so
    any run reports the same spec. Figures that recompute the fiducial spectrum
    need it to know how many samples of the bank the injection actually draws.
    """
    catalog = load_run_config(*run).get("catalog") or {}
    injection = catalog.get("injection")
    if not injection:
        raise ValueError(f"{_label(run)} must define a [catalog.injection] table")
    return CatalogSpec.model_validate(injection)


def load_proposal_spec(run: tuple[str, str] = REFERENCE_RUN) -> CatalogSpec:
    """The proposal catalog an assembled run composes.

    Unlike the injection, this one differs between experiments -- the default
    is what the reference run uses.
    """
    catalog = load_run_config(*run).get("catalog") or {}
    proposal = catalog.get("proposal")
    if not proposal:
        raise ValueError(f"{_label(run)} must define a [catalog.proposal] table")
    return CatalogSpec.model_validate(proposal)


def reference_config_path(run: tuple[str, str] = REFERENCE_RUN) -> Path:
    """The assembled config a figure rule should declare as an input."""
    return config_path(*run)


def _label(run: tuple[str, str]) -> str:
    return f"{run[0]}/{run[1]}"
