"""An ensemble of spectrum draws, cached one file per draw.

Several analyses draw the same record many times -- a mean spectrum, its
sample variance, coverage injections -- and a Poisson draw at the paper's
rate reduces several hundred thousand waveforms. :func:`spectra_ensemble`
draws each one once: draw ``i`` of ``seed`` is
``simulator(batch_keys(seed, n)[i])``, and because those keys are
prefix-stable it does not depend on ``n``. So every caller asking for the same
record and seed shares the same files, whatever ensemble size each asks for,
and an interrupted run resumes where it stopped. ``start`` draws a later slice
alone, so machines sharing one cache directory can each draw a disjoint range
of indices and the files merge by copying.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from astrogwb.simulators.core import batch_keys, load, write
from astrogwb.simulators.spectra.metadata import BackgroundSpectralDensityMetadata
from astrogwb.simulators.spectra.network import (
    JointNetworkSpectralDensitySimulator,
    NetworkSpectralDensityMetadata,
)
from astrogwb.simulators.spectra.simulator import (
    BackgroundSpectralDensityData,
    BackgroundSpectralDensitySimulator,
    stack_spectra,
)

__all__ = ["network_spectra_ensemble", "spectra_ensemble", "spectra_ensemble_path"]


def spectra_ensemble_path(
    directory: str | Path,
    metadata: BackgroundSpectralDensityMetadata,
    seed: int,
    index: int,
) -> Path:
    """Where draw ``index`` of ``seed`` lives: ``<directory>/<key>/<seed>-<index>.h5``."""
    return Path(directory) / metadata.key() / f"{int(seed)}-{int(index)}.h5"


def spectra_ensemble(
    metadata: BackgroundSpectralDensityMetadata,
    seed: int,
    num_draws: int,
    directory: str | Path,
    *,
    start: int = 0,
    chunk_size: int = 128,
    source_chunk_size: int | None = None,
) -> BackgroundSpectralDensityData:
    """Draws ``start .. num_draws - 1`` of ``metadata`` at ``seed``, stacked.

    A missing draw is simulated and written with :func:`~astrogwb.simulators.core.write`,
    which records the metadata and seed beside the data; a cached one is read
    back and checked against both. The simulator is built only if a draw is
    missing. ``chunk_size`` and ``source_chunk_size`` are
    :class:`~astrogwb.simulators.spectra.BackgroundSpectralDensitySimulator`'s
    cost settings: they change no draw.

    Parameters
    ----------
    metadata:
        The record to draw.
    seed:
        Draw ``i`` uses ``batch_keys(seed, num_draws)[i]``.
    num_draws:
        Ensemble size, one past the last index; a smaller one reads a prefix
        of a larger one's files.
    directory:
        Cache root; see :func:`spectra_ensemble_path`.
    start:
        The first index, ``0 <= start < num_draws``. Draw ``i`` is the same
        whatever ``start`` is.

    Returns
    -------
    BackgroundSpectralDensityData
        The draws stacked along the leading axis, in index order.

    Raises
    ------
    ValueError
        If a cached file records another metadata or seed.
    """
    keys = batch_keys(seed, num_draws)
    simulator: BackgroundSpectralDensitySimulator | None = None
    parts: list[BackgroundSpectralDensityData] = []
    for index in range(start, num_draws):
        path = spectra_ensemble_path(directory, metadata, seed, index)
        if path.is_file():
            parts.append(_load_draw(path, metadata, seed))
            continue
        if simulator is None:
            simulator = BackgroundSpectralDensitySimulator(
                metadata, chunk_size=chunk_size, source_chunk_size=source_chunk_size
            )
        drawn = simulator(keys[index])
        write(path, drawn, metadata, seed=seed)
        parts.append(drawn)
    return _stack(parts)


def network_spectra_ensemble(
    metadata: Sequence[NetworkSpectralDensityMetadata],
    seed: int,
    num_draws: int,
    directory: str | Path,
    *,
    start: int = 0,
    chunk_size: int = 128,
    source_chunk_size: int | None = None,
) -> list[BackgroundSpectralDensityData]:
    """:func:`spectra_ensemble` of several networks, drawn jointly.

    Each network's draws are cached under its own key, exactly where
    :func:`spectra_ensemble_path` puts them, so one network can be read back
    alone. A draw missing for any network is simulated for all of them with
    :class:`~astrogwb.simulators.spectra.JointNetworkSpectralDensitySimulator`,
    whose contract on ``metadata`` this shares: one waveform pass per draw,
    whatever the number of networks. Draw ``i`` of every network uses
    ``batch_keys(seed, num_draws)[i]``, so the networks see the same sources.

    Parameters
    ----------
    metadata:
        One record per network, differing only in the network.
    seed, num_draws, directory, start, chunk_size, source_chunk_size:
        As for :func:`spectra_ensemble`.

    Returns
    -------
    list of BackgroundSpectralDensityData
        Each network's draws stacked along the leading axis, in ``metadata``
        order.

    Raises
    ------
    ValueError
        If a cached file records another metadata or seed.
    """
    keys = batch_keys(seed, num_draws)
    simulator: JointNetworkSpectralDensitySimulator | None = None
    parts: list[list[BackgroundSpectralDensityData]] = [[] for _ in metadata]
    for index in range(start, num_draws):
        paths = [
            spectra_ensemble_path(directory, record, seed, index) for record in metadata
        ]
        if all(path.is_file() for path in paths):
            for part, path, record in zip(parts, paths, metadata, strict=True):
                part.append(_load_draw(path, record, seed))
            continue
        if simulator is None:
            simulator = JointNetworkSpectralDensitySimulator(
                metadata, chunk_size=chunk_size, source_chunk_size=source_chunk_size
            )
        for part, path, record, drawn in zip(
            parts, paths, metadata, simulator(keys[index]), strict=True
        ):
            write(path, drawn, record, seed=seed)
            part.append(drawn)
    return [_stack(part) for part in parts]


def _load_draw(
    path: Path, metadata: BackgroundSpectralDensityMetadata, seed: int
) -> BackgroundSpectralDensityData:
    """A cached draw, checked against the record and seed it was asked for."""
    data, recorded, attrs = load(path, type(metadata))
    if recorded.key() != metadata.key() or int(attrs["seed"]) != seed:
        raise ValueError(
            f"{path} records {recorded.key()} at seed {attrs.get('seed')}, "
            f"not the requested {metadata.key()} at seed {seed}"
        )
    return data  # ty: ignore[invalid-return-type]


def _stack(parts: list[BackgroundSpectralDensityData]) -> BackgroundSpectralDensityData:
    """:func:`stack_spectra`, with the spectral density as float64."""
    stacked = stack_spectra(parts)
    stacked["spectral_density"] = np.asarray(
        stacked["spectral_density"], dtype=np.float64
    )
    return stacked
