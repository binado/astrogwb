"""Read an ensemble of simulated backgrounds back, stacked per network.

``astrogwb simulate`` writes one file per draw, ``<root>/<seed>-<index>.h5``,
each a :class:`~astrogwb.background.simulation.BackgroundDraw` stored by
:meth:`~astrogwb.background.simulation.BackgroundDraw.write`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from astrogwb.background.simulation.draw import BackgroundDraw

__all__ = ["NetworkSpectra", "network_spectra_ensemble"]


class NetworkSpectra(TypedDict):
    """One network's draws, stacked along the leading draw axis.

    ``frequencies`` ``(F,)``; ``pair_spectral_density`` ``(D, P, F)``, which
    :func:`~astrogwb.statistics.cross_correlation_estimator` weights into a network
    spectrum ``(D, F)``; ``n_events`` ``(D,)``, the number of events each draw
    selected.
    """

    frequencies: NDArray[np.float64]
    pair_spectral_density: NDArray[np.float64]
    n_events: NDArray[np.int64]


def network_spectra_ensemble(
    root: str | Path,
    seed: int,
    num_draws: int,
    names: Sequence[str],
    *,
    start: int = 0,
) -> list[NetworkSpectra]:
    """Draws ``start .. num_draws - 1`` of the networks ``names``, stacked per network.

    Draw ``i`` is the background file ``<root>/<seed>-<i>.h5``, simulated at
    ``realization_key(seed, i)``.

    Parameters
    ----------
    root
        The ensemble root.
    seed
        The seed the ensemble was simulated with.
    num_draws
        One past the last index read.
    names
        Networks, each recorded in every draw's background.
    start
        The first index read; must be below ``num_draws``. Every draw is
        taken to share one frequency grid.

    Returns
    -------
    list of NetworkSpectra
        Each network's draws, in ``names`` order.

    Raises
    ------
    FileNotFoundError
        If a draw's file is missing.
    KeyError
        If a draw's background does not record one of ``names``.
    """
    root = Path(root)
    draws = [
        BackgroundDraw.read(root / f"{int(seed)}-{index}.h5")[0]
        for index in range(start, num_draws)
    ]
    return [_stack(draws, name) for name in names]


def _stack(draws: Sequence[BackgroundDraw], name: str) -> NetworkSpectra:
    """One network's spectra of every draw, stacked."""
    return {
        "frequencies": draws[0].frequencies,
        "pair_spectral_density": np.stack([draw.spectra[name] for draw in draws]),
        "n_events": np.asarray(
            [draw.segment_counts.sum() for draw in draws], dtype=np.int64
        ),
    }
