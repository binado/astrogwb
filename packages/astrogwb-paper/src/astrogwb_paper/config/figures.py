"""Configuration the paper figure scripts read for themselves (stdlib only).

Presentation metadata -- ordered run names, LaTeX labels, and the shared
detector-network label registry -- lives under ``inputs/figures/``. Detector
*lists* are deliberately not duplicated here: they stay in
``experiments/<name>.toml`` and are resolved through
:func:`astrogwb_paper.config.experiments.overlay_for`, so a network's name,
label, and detectors always travel together as one :class:`Network` instead of
as parallel argv lists kept aligned by hand.

Scientific values shared by every run (fiducials, frequency band, redshift
grid) are read straight from ``inputs/mcmc.base.toml``, the same file the MCMC
runs are assembled from.

Like :mod:`astrogwb_paper.config.loading`, this module imports neither JAX nor
``astrogwb``, so a figure script can parse and validate its configuration
before touching a runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrogwb_paper.config.experiments import experiment, overlay_for
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.paths import resolve_paper_path

NETWORK_LABELS_PATH = Path("inputs/figures/detector-networks.toml")


@dataclass(frozen=True)
class Network:
    """One detector network: its run name, LaTeX label, and detector list."""

    name: str
    label: str
    detectors: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisGrid:
    """Frequency band and redshift grid shared by every experiment run."""

    observation_time: float
    f_min: float
    f_max: float
    z_min: float
    z_max: float
    n_grid: int


def load_figure_config(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Load one ``inputs/figures/*.toml`` presentation config."""
    return load_mapping(resolve_paper_path(path, root))


def load_network_labels(
    path: Path = NETWORK_LABELS_PATH, root: Path | None = None
) -> dict[str, str]:
    """Load the shared detector-network label registry."""
    resolved = resolve_paper_path(path, root)
    labels = load_mapping(resolved).get("labels")
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError(f"{resolved} must define a non-empty [labels] table")
    return {str(name): str(label) for name, label in labels.items()}


def resolve_networks(
    experiment_name: str,
    run_names: Sequence[str],
    labels: Mapping[str, str],
) -> tuple[Network, ...]:
    """Pair each run with its registry label and the experiment's detectors.

    Declaration order is preserved: it drives chain order, legend order, and
    the color/linestyle assignment in the detector-comparison figures.
    """
    if not run_names:
        raise ValueError(f"{experiment_name} figure declares no detector networks")
    duplicates = sorted({run for run in run_names if run_names.count(run) > 1})
    if duplicates:
        raise ValueError("duplicate detector network(s): " + ", ".join(duplicates))
    unlabelled = [run for run in run_names if run not in labels]
    if unlabelled:
        raise ValueError(
            "detector-network label registry has no entry for: " + ", ".join(unlabelled)
        )

    spec = experiment(experiment_name)
    networks: list[Network] = []
    for run in run_names:
        detectors = overlay_for(spec, run).get("analysis", {}).get("detectors")
        if not detectors:
            raise ValueError(f"{experiment_name}/{run} declares no analysis.detectors")
        networks.append(Network(run, labels[run], tuple(detectors)))
    return tuple(networks)


def figure_networks(
    figure: Mapping[str, Any], key: str, root: Path | None = None
) -> tuple[Network, ...]:
    """Resolve the ordered run-name array ``key`` of a loaded figure config."""
    return resolve_networks(
        figure["experiment"], figure[key], load_network_labels(root=root)
    )


def load_fiducials(path: Path, root: Path | None = None) -> dict[str, float]:
    """Load the shared ``[fiducials]`` table from the base MCMC config."""
    resolved = resolve_paper_path(path, root)
    fiducials = load_mapping(resolved).get("fiducials")
    if not isinstance(fiducials, Mapping) or not fiducials:
        raise ValueError(f"{resolved} must define a non-empty [fiducials] table")
    return {str(name): float(value) for name, value in fiducials.items()}


def load_analysis_grid(path: Path, root: Path | None = None) -> AnalysisGrid:
    """Load the shared frequency band and redshift grid from the base config."""
    resolved = resolve_paper_path(path, root)
    base = load_mapping(resolved)
    analysis = base.get("analysis") or {}
    cosmology = base.get("cosmology") or {}
    try:
        return AnalysisGrid(
            observation_time=float(base["observation_time"]),
            f_min=float(analysis["f_min"]),
            f_max=float(analysis["f_max"]),
            z_min=float(cosmology["z_min"]),
            z_max=float(cosmology["z_max"]),
            n_grid=int(cosmology["n_grid"]),
        )
    except KeyError as error:
        raise ValueError(f"{resolved} is missing analysis setting {error}") from None
