"""The frequency band and redshift grid every analysis input is built on.

A stdlib-only leaf so both sides of the config graph can depend on it. The
figure path reaches it through :mod:`astrogwb_paper.config.figures` (which
reads it back out of an assembled run config) and the sampling path through
:attr:`astrogwb_paper.config.mcmc.RunConfig.analysis_grid` (which assembles one
from an already-validated run config). ``config.mcmc`` must not import
``config.figures``: the graph runs figures -> runs -> mcmc -> loading, so that
edge would be a cycle and would drag the filesystem-reading discovery layer
into the ``RunConfig`` path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisGrid:
    """Frequency band and redshift grid shared by every experiment run."""

    observation_time: float
    f_min: float
    f_max: float
    minimum_redshift: float
    maximum_redshift: float
    n_grid: int
