"""Scientific inputs the paper figure scripts read for themselves (JAX-free).

Presentation -- which runs a figure shows, in what order, under which LaTeX
label -- is hard-coded in the figure scripts and in
:mod:`astrogwb_paper.plotting`. It is not configuration: changing a legend
label is a code change, reviewed with the plot it labels.

What stays here is the part with scientific consequences. Detector *lists* are
never restated alongside a label: they live in ``inputs/experiments.yaml`` under
``experiments.<name>.runs.<run>.analysis.detectors`` and are attached by
:func:`resolve_networks`, so the detectors a figure computes an SNR for are
always the ones its chain was sampled with. Fiducials, the frequency band, and
the redshift grid are read from that inventory's ``base`` mapping.

Like :mod:`astrogwb_paper.config.loading`, this module imports neither JAX nor
``astrogwb``, so a figure script can resolve and validate its inputs before
touching a runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from astrogwb_paper.config.analysis import AnalysisGrid
from astrogwb_paper.config.experiments import (
    experiment,
    inventory_path,
    load_base,
    overlay_for,
)


@dataclass(frozen=True)
class Network:
    """One detector network: its run name, LaTeX label, and detector list."""

    name: str
    label: str
    detectors: tuple[str, ...]


def resolve_networks(
    experiment_name: str,
    networks: Sequence[tuple[str, str]],
) -> tuple[Network, ...]:
    """Attach each ``(run, label)`` pair's detectors from its experiment.

    Declaration order is preserved: it drives chain order, legend order, and
    the color/linestyle assignment in the detector-comparison figures.
    """
    if not networks:
        raise ValueError(f"{experiment_name} figure declares no detector networks")
    run_names = [name for name, _ in networks]
    duplicates = sorted({run for run in run_names if run_names.count(run) > 1})
    if duplicates:
        raise ValueError("duplicate detector network(s): " + ", ".join(duplicates))

    spec = experiment(experiment_name)
    resolved: list[Network] = []
    for name, label in networks:
        detectors = overlay_for(spec, name).get("analysis", {}).get("detectors")
        if not detectors:
            raise ValueError(f"{experiment_name}/{name} declares no analysis.detectors")
        resolved.append(Network(name, label, tuple(detectors)))
    return tuple(resolved)


def load_fiducials() -> dict[str, float]:
    """Load the shared ``fiducials`` mapping from the MCMC inventory."""
    fiducials = load_base().get("fiducials")
    if not isinstance(fiducials, Mapping) or not fiducials:
        raise ValueError(
            f"{inventory_path()} must define a non-empty [fiducials] table"
        )
    return {str(name): float(value) for name, value in fiducials.items()}


def load_analysis_grid() -> AnalysisGrid:
    """Load the shared frequency band and redshift grid from the inventory."""
    base = load_base()
    analysis = base.get("analysis") or {}
    cosmology = base.get("cosmology") or {}
    try:
        return AnalysisGrid(
            observation_time=float(base["observation_time"]),
            f_min=float(analysis["f_min"]),
            f_max=float(analysis["f_max"]),
            minimum_redshift=float(cosmology["minimum_redshift"]),
            maximum_redshift=float(cosmology["maximum_redshift"]),
            n_grid=int(cosmology["n_grid"]),
        )
    except KeyError as error:
        raise ValueError(
            f"{inventory_path()} is missing analysis setting {error}"
        ) from None
