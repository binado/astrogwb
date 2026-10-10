"""An ensemble of spectrum draws, cached one file per draw.

Several analyses draw the same record many times -- a mean spectrum, its
sample variance, coverage injections -- and a Poisson draw at the paper's
rate reduces several hundred thousand waveforms. :func:`spectra_ensemble`
draws each one once: draw ``i`` of ``seed`` is
``simulator(batch_keys(seed, n)[i])``, and because those keys are
prefix-stable it does not depend on ``n``. So every caller asking for the same
record and seed shares the same files, whatever ensemble size each asks for,
and an interrupted run resumes where it stopped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astrogwb.simulators.core import batch_keys, load, write
from astrogwb.simulators.spectra.metadata import BackgroundSpectralDensityMetadata
from astrogwb.simulators.spectra.simulator import (
    BackgroundSpectralDensityData,
    BackgroundSpectralDensitySimulator,
    stack_spectra,
)

__all__ = ["spectra_ensemble", "spectra_ensemble_path"]


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
    chunk_size: int = 128,
    source_chunk_size: int | None = None,
) -> BackgroundSpectralDensityData:
    """Draws ``0 .. num_draws - 1`` of ``metadata`` at ``seed``, stacked.

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
        Ensemble size; a smaller one reads a prefix of a larger one's files.
    directory:
        Cache root; see :func:`spectra_ensemble_path`.

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
    for index in range(num_draws):
        path = spectra_ensemble_path(directory, metadata, seed, index)
        if path.is_file():
            data, recorded, attrs = load(path, BackgroundSpectralDensityMetadata)
            if recorded.key() != metadata.key() or int(attrs["seed"]) != seed:
                raise ValueError(
                    f"{path} records {recorded.key()} at seed {attrs.get('seed')}, "
                    f"not the requested {metadata.key()} at seed {seed}"
                )
            parts.append(data)  # ty: ignore[invalid-argument-type]
            continue
        if simulator is None:
            simulator = BackgroundSpectralDensitySimulator(
                metadata, chunk_size=chunk_size, source_chunk_size=source_chunk_size
            )
        drawn = simulator(keys[index])
        write(path, drawn, metadata, seed=seed)
        parts.append(drawn)
    stacked = stack_spectra(parts)
    stacked["spectral_density"] = np.asarray(
        stacked["spectral_density"], dtype=np.float64
    )
    return stacked
