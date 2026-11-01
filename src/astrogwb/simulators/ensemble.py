"""Read an ensemble of simulated backgrounds back, stacked per network.

``workflow/coverage_draws.smk`` writes one file per draw,
``<root>/<seed>-<index>.h5``, each a :class:`BackgroundArtifact`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from astrogwb.simulators.background import BackgroundArtifact

__all__ = ["NetworkSpectra", "network_spectra_ensemble"]


class NetworkSpectra(TypedDict):
    """One network's draws, stacked along the leading draw axis.

    ``frequencies`` ``(F,)``; ``pair_spectral_density`` ``(D, P, F)``, which
    :func:`~astrogwb.simulators.combine_pair_spectra` weights into a network
    spectrum ``(D, F)``; ``n_events`` and
    ``total_merger_rate`` ``(D,)``; each ``hyperparameters`` value ``(D,)``.
    ``n_events`` is each draw's ``count``, named as the Poisson-count model's
    site is.
    """

    frequencies: NDArray[np.float64]
    pair_spectral_density: NDArray[np.float64]
    n_events: NDArray[np.int64]
    total_merger_rate: NDArray[np.float64]
    hyperparameters: dict[str, NDArray[np.float64]]


def network_spectra_ensemble(
    root: str | Path,
    seed: int,
    num_draws: int,
    names: Sequence[str],
    *,
    start: int = 0,
) -> list[NetworkSpectra]:
    """Draws ``start .. num_draws - 1`` of the networks ``names``, stacked per network.

    Draw ``i`` is the background file ``<root>/<seed>-<i>.h5``,
    as ``workflow/coverage_draws.smk`` lays them out, draw ``i`` keyed by
    ``batch_keys(seed, i + 1)[i]``.

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
        taken to share one frequency grid and one set of hyperparameters.

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
        BackgroundArtifact.open(root / f"{int(seed)}-{index}.h5")
        for index in range(start, num_draws)
    ]
    return [_stack(draws, name) for name in names]


def _stack(draws: Sequence[BackgroundArtifact], name: str) -> NetworkSpectra:
    """One network's spectra of every draw, stacked."""
    return {
        "frequencies": draws[0].frequencies,
        "pair_spectral_density": np.stack([draw.spectra[name] for draw in draws]),
        "n_events": np.asarray([draw.count for draw in draws], dtype=np.int64),
        "total_merger_rate": np.asarray(
            [draw.total_merger_rate for draw in draws], dtype=np.float64
        ),
        "hyperparameters": {
            key: np.asarray([draw.hyperparameters[key] for draw in draws])
            for key in draws[0].hyperparameters
        },
    }
