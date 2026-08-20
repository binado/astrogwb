"""The frequency band and redshift grid every analysis input is built on.

A stdlib-only leaf so both sides of the config graph can depend on it. The
figure path reaches it through :mod:`astrogwb_paper.config.figures` (which
reads it from the experiment inventory) and the sampling path through
:attr:`astrogwb_paper.config.mcmc.RunConfig.analysis_grid` (which assembles
one from an already-validated run config). ``config.mcmc`` must not import
``config.figures``: the graph runs figures -> experiments -> catalogs ->
loading, so that edge would drag the YAML inventory readers into the
``RunConfig`` path and break its stdlib+pydantic-only guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisGrid:
    """Frequency band and redshift grid shared by every experiment run."""

    observation_time: float
    f_min: float
    f_max: float
    z_min: float
    z_max: float
    n_grid: int
