"""Poisson realizations of the background, in two stages.

1. :class:`PopulationSimulator` draws one realization -- hyperparameters, a
   Poisson count, every source's intrinsic and extrinsic parameters -- as a
   :class:`PopulationArtifact`.
2. :class:`BackgroundSimulator` generates each chunk's radiation-frame
   polarizations through a :class:`~astrogwb.waveform.WaveformMetadata`,
   projects them onto the named networks of a
   :class:`~astrogwb.detector.DetectorRegistry`, and sums the per-pair
   cross-powers into a :class:`BackgroundArtifact` holding each network's pair
   spectra and their :func:`combine_pair_spectra`.

:func:`detector_noise` evaluates the networks' noise on a background's
frequencies once per ensemble, a :class:`DetectorNoise`.

Fused in memory, ``BackgroundSimulator(...)(population)`` never holds more than
one chunk of polarizations. On disk, a realization is one directory, a
:class:`Realization`: one ``metadata.toml`` and at most one product per stage,
which :class:`RealizationSimulator` adds to stage by stage;
:func:`network_spectra_ensemble` reads an ensemble of realizations, one
directory per seed and index, back stacked per network. Only the population
stage draws random numbers.
"""

from astrogwb.simulators._io import batch_keys, realization_key
from astrogwb.simulators.background import (
    BackgroundArtifact,
    BackgroundSimulator,
    NetworkSpectrum,
    combine_pair_spectra,
    waveform_key,
)
from astrogwb.simulators.noise import (
    DetectorNoise,
    DetectorNoiseMetadata,
    NetworkNoise,
    detector_noise,
)
from astrogwb.simulators.population import (
    PopulationArtifact,
    PopulationMetadata,
    PopulationSimulator,
    population_key,
)
from astrogwb.simulators.realization import (
    NetworkSpectra,
    Realization,
    RealizationSimulator,
    network_spectra_ensemble,
    simulate_realization,
)

__all__ = [
    "BackgroundArtifact",
    "BackgroundSimulator",
    "DetectorNoise",
    "DetectorNoiseMetadata",
    "NetworkNoise",
    "NetworkSpectra",
    "NetworkSpectrum",
    "PopulationArtifact",
    "PopulationMetadata",
    "PopulationSimulator",
    "Realization",
    "RealizationSimulator",
    "batch_keys",
    "combine_pair_spectra",
    "detector_noise",
    "network_spectra_ensemble",
    "population_key",
    "realization_key",
    "simulate_realization",
    "waveform_key",
]
