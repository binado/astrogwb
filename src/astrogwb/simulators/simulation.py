"""One-pass simulation: a configuration and a key in, a background out."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrogwb.simulators.background import BackgroundArtifact, BackgroundSimulator
from astrogwb.simulators.config import SimulationConfig
from astrogwb.simulators.population import PopulationSimulator

if TYPE_CHECKING:
    import jax

__all__ = ["SimulationConfig", "simulate"]


def simulate(
    config: SimulationConfig,
    key: jax.Array,
    *,
    chunk_size: int = 1024,
    segment_batch_size: int = 1024,
) -> BackgroundArtifact:
    """Simulate ``config`` at ``key``: draw the population, then its background.

    Parameters
    ----------
    config
        The simulation configuration.
    key
        A single typed JAX key, e.g. from
        :func:`~astrogwb.simulators.realization_key`.
    chunk_size
        Waveforms evaluated per chunk.
    segment_batch_size
        Noise segments whose strains are held at once. Both only change cost,
        never the result.

    Returns
    -------
    BackgroundArtifact
        The pair spectra, carrying ``config``.
    """
    population = PopulationSimulator(config.population)(key)
    return BackgroundSimulator(
        config.waveform,
        config.detectors,
        chunk_size=chunk_size,
        segment_duration=config.segment_duration,
        segment_batch_size=segment_batch_size,
    )(population)
