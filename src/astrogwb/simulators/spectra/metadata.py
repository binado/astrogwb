"""Everything that determines a set of forward-model spectral-density draws."""

from __future__ import annotations

from astrogwb.simulators.population.metadata import (
    Hyperparameter,
    PopulationDrawMetadata,
)
from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["BackgroundSpectralDensityMetadata", "Hyperparameter"]


class BackgroundSpectralDensityMetadata(PopulationDrawMetadata):
    """The record a spectral-density artifact is generated from and cached under.

    A spectrum is a population draw reduced through a waveform, so this is a
    :class:`~astrogwb.simulators.population.PopulationDrawMetadata` plus the
    ``waveform`` that reduces it; :attr:`sources` is the waveform-free part,
    which names the population draw a spectrum was made from. It is both the
    request and the provenance: a simulator turns it into draws, the artifact
    carries it, and :meth:`key` is the file name the draws are cached under.
    Nothing outside it changes what the draws contain.

    Neither a seed nor a draw count is part of it: the seeds are the
    simulator's input, one per draw. ``count`` selects Poisson counts or exactly
    ``num_events`` sources per realization, both exact -- there is no padded
    capacity. ``observation_time`` is positive in both modes: it determines
    Poisson counts, but cancels from fixed-count spectrum normalization.
    ``chunk_size`` is not part of it: it only shapes the reduction and consumes
    no randomness, so it is the simulator's.
    """

    waveform: WaveformMetadata

    @property
    def sources(self) -> PopulationDrawMetadata:
        """The population draw this spectrum reduces, without its waveform."""
        return PopulationDrawMetadata.model_validate(
            {name: getattr(self, name) for name in PopulationDrawMetadata.model_fields}
        )
