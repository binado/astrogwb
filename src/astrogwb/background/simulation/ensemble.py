"""Read background draws and stack their pair spectra by detector network."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from astrogwb.background.simulation.draw import BackgroundDraw

__all__ = ["read_background_ensemble"]


class _NetworkSpectra(TypedDict):
    """One network's draws, stacked along the leading draw axis.

    ``frequencies`` ``(F,)``; ``pair_spectral_density`` ``(D, P, F)``, which
    :func:`~astrogwb.statistics.cross_correlation_estimator` weights into a network
    spectrum ``(D, F)``; ``n_events`` ``(D,)``, the number of events each draw
    selected.
    """

    frequencies: NDArray[np.float64]
    pair_spectral_density: NDArray[np.float64]
    n_events: NDArray[np.int64]


def read_background_ensemble(
    paths: Sequence[str | Path], names: Sequence[str]
) -> dict[str, _NetworkSpectra]:
    """Read draws and stack their spectra and event counts in path order.

    Parameters
    ----------
    paths
        Background draw files on a shared frequency grid, in ensemble order.
    names
        Network names present in the draws.

    Returns
    -------
    dict
        For each network, its frequencies, pair spectral densities stacked
        along the leading draw axis and total event count per draw.
    """
    draws = [BackgroundDraw.read(path)[0] for path in paths]
    return {name: _stack(draws, name) for name in names}


def _stack(draws: Sequence[BackgroundDraw], name: str) -> _NetworkSpectra:
    """One network's spectra of every draw, stacked."""
    return {
        "frequencies": draws[0].frequencies,
        "pair_spectral_density": np.stack([draw.spectra[name] for draw in draws]),
        "n_events": np.asarray(
            [draw.segment_counts.sum() for draw in draws], dtype=np.int64
        ),
    }
