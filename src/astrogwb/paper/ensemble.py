"""Ensembles of network realizations, cached one realization directory per draw.

Several analyses draw the same record many times -- a mean spectrum, its
sample variance, coverage injections -- and a Poisson draw at the paper's rate
projects several hundred thousand waveforms. :func:`network_spectra_ensemble`
draws each one once, through the fused stages of :mod:`astrogwb.simulators`:
the population and its polarizations stay in memory, chunk by chunk, and only
the manifest and the per-pair background of each realization are written. Draw ``i`` of ``seed`` is the realization
at ``batch_keys(seed, n)[i]``, and those keys are prefix-stable, so it does
not depend on ``n``: every caller asking for the same record and seed shares
the same files, whatever ensemble size each asks for, an interrupted run
resumes where it stopped, and machines sharing a cache can each draw a
disjoint range of indices and merge the files by copying.

All networks share one background over the union of their detectors, so the
waveforms -- the expensive half -- are generated once per draw, every network
sees the same sources, and a pair two networks share is projected once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from astrogwb.paper.config.detectors import DetectorRegistry
from astrogwb.simulators import (
    BackgroundArtifact,
    DetectorConfiguration,
    PopulationMetadata,
    RealizationSimulator,
    combine_pair_spectra,
    simulate_ensemble,
    union_configuration,
)
from astrogwb.utils.io import content_key
from astrogwb.waveform import WaveformMetadata

__all__ = [
    "EnsembleRecord",
    "NetworkSpectra",
    "ensemble_path",
    "network_configuration",
    "network_spectra_ensemble",
]


class NetworkSpectra(TypedDict):
    """One network's draws, stacked along the leading draw axis.

    ``frequencies`` ``(F,)``; ``spectral_density`` ``(D, F)``, the
    :func:`~astrogwb.simulators.combine_pair_spectra` of
    ``pair_spectral_density`` ``(D, P, F)``; ``n_events`` and
    ``total_merger_rate`` ``(D,)``; each ``hyperparameters`` value ``(D,)``.
    """

    frequencies: NDArray[np.float64]
    spectral_density: NDArray[np.float64]
    pair_spectral_density: NDArray[np.float64]
    n_events: NDArray[np.int64]
    total_merger_rate: NDArray[np.float64]
    hyperparameters: dict[str, NDArray[np.float64]]


def network_configuration(
    registry: DetectorRegistry, name: str
) -> DetectorConfiguration:
    """The network ``name`` of ``registry``, in declared order, with its noise curves.

    Detectors are named by site code, which resolves to the packaged geometry:
    a registry geometry override is not carried into it.
    """
    detectors = list(registry.networks[name])
    return DetectorConfiguration(
        detectors=detectors,
        psd_references={
            detector: registry.detectors[detector].psd_reference
            for detector in detectors
        },
    )


@dataclass(frozen=True)
class EnsembleRecord:
    """What an ensemble draws: a population, a waveform and the networks recording it.

    ``networks`` maps a name to its configuration, in output order. The draws
    are cached under :meth:`key`, which names the population, the waveform
    and the union of the networks' detectors -- not the noise curves, which
    only the network spectra depend on.
    """

    population: PopulationMetadata
    waveform: WaveformMetadata
    networks: Mapping[str, DetectorConfiguration]

    def key(self) -> str:
        """The content hash the draws are cached under; seeds aside."""
        return content_key(
            {
                "population": self.population.payload(),
                "waveform": self.waveform.model_dump(mode="json"),
                "detectors": union_configuration(
                    list(self.networks.values())
                ).detectors,
            }
        )


def ensemble_path(directory: str | Path, key: str, seed: int, index: int) -> Path:
    """Where draw ``index`` of ``seed`` lives: ``<directory>/<key>/<seed>-<index>``."""
    return Path(directory) / key / f"{int(seed)}-{int(index)}"


def network_spectra_ensemble(
    record: EnsembleRecord,
    seed: int,
    num_draws: int,
    directory: str | Path,
    *,
    start: int = 0,
    chunk_size: int = 1024,
    source_chunk_size: int = 65_536,
) -> list[NetworkSpectra]:
    """Draws ``start .. num_draws - 1`` of every network, stacked per network.

    Each draw is a realization directory of
    :func:`~astrogwb.simulators.simulate_ensemble`: a missing one is simulated
    and written with its background only, a cached one is read after its
    population key is checked against the record and seed. The simulators are
    built only if a draw is missing. ``chunk_size`` (waveforms per chunk) and
    ``source_chunk_size`` (sources drawn per piece) are cost settings: they
    change no draw.

    Parameters
    ----------
    record:
        The realizations to draw and the networks recording them.
    seed:
        Draw ``i`` uses ``batch_keys(seed, num_draws)[i]``.
    num_draws:
        Ensemble size, one past the last index; a smaller one reads a prefix
        of a larger one's files.
    directory:
        Cache root; see :func:`ensemble_path`.
    start:
        The first index, ``0 <= start < num_draws``.

    Returns
    -------
    list of NetworkSpectra
        Each network's draws, in ``record.networks`` order.

    Raises
    ------
    ValueError
        If a cached draw records another population or realization.
    """
    networks = list(record.networks.values())
    simulator = RealizationSimulator(
        record.population,
        record.waveform,
        union_configuration(networks),
        keep_sources=False,
        chunk_size=chunk_size,
        source_chunk_size=source_chunk_size,
    )
    realizations = simulate_ensemble(
        simulator, Path(directory) / record.key(), seed, num_draws, start=start
    )
    draws = [realization.background() for realization in realizations]
    return [_stack(draws, network) for network in networks]


def _stack(
    draws: Sequence[BackgroundArtifact], network: DetectorConfiguration
) -> NetworkSpectra:
    """One network's rows of every draw, stacked, and their network spectrum."""
    frequencies = draws[0].frequencies
    pairs = np.stack([draw.pair_spectra(network) for draw in draws])
    return {
        "frequencies": frequencies,
        "spectral_density": combine_pair_spectra(pairs, frequencies, network),
        "pair_spectral_density": pairs,
        "n_events": np.asarray([draw.count for draw in draws], dtype=np.int64),
        "total_merger_rate": np.asarray(
            [draw.total_merger_rate for draw in draws], dtype=np.float64
        ),
        "hyperparameters": {
            name: np.asarray([draw.hyperparameters[name] for draw in draws])
            for name in draws[0].hyperparameters
        },
    }
