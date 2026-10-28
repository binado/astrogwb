"""Poisson realizations of the background, in three stages.

1. :class:`PopulationSimulator` draws one realization -- hyperparameters, a
   Poisson count, every source's intrinsic and extrinsic parameters -- as a
   :class:`PopulationArtifact`.
2. :class:`WaveformSimulator` turns it into complex radiation-frame
   polarizations, :class:`Waveforms`, generated chunk by chunk; written, they
   are a :class:`WaveformArtifact`.
3. :class:`BackgroundSimulator` projects them onto named networks, a
   :data:`NetworkConfiguration` of :class:`DetectorConfiguration`, and sums
   the per-pair cross-powers into a :class:`BackgroundArtifact` holding each
   network's pair spectra and their :func:`combine_pair_spectra`.

:func:`detector_noise` evaluates the networks' noise on a background's
frequencies once per ensemble, a :class:`DetectorNoise`.

Fused in memory, ``BackgroundSimulator(...)(WaveformSimulator(...)(
population))`` never holds more than one chunk of polarizations. On disk, a
realization is one directory, a :class:`Realization`: one ``metadata.toml``
and at most one product per stage, which :class:`RealizationSimulator` adds to
stage by stage, and :func:`simulate_ensemble` lays out by seed and index;
:func:`network_spectra_ensemble` reads such an ensemble back, stacked per
network. Only the population stage draws random numbers.
"""

from astrogwb.simulators._io import batch_keys, realization_key
from astrogwb.simulators.background import (
    BackgroundArtifact,
    BackgroundSimulator,
    DetectorConfiguration,
    NetworkConfiguration,
    NetworkSpectrum,
    combine_pair_spectra,
    union_configuration,
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
    simulate_ensemble,
    simulate_realization,
)
from astrogwb.simulators.waveforms import (
    WaveformArtifact,
    Waveforms,
    WaveformSimulator,
)

__all__ = [
    "BackgroundArtifact",
    "BackgroundSimulator",
    "DetectorConfiguration",
    "DetectorNoise",
    "DetectorNoiseMetadata",
    "NetworkConfiguration",
    "NetworkNoise",
    "NetworkSpectra",
    "NetworkSpectrum",
    "PopulationArtifact",
    "PopulationMetadata",
    "PopulationSimulator",
    "Realization",
    "RealizationSimulator",
    "WaveformArtifact",
    "WaveformSimulator",
    "Waveforms",
    "batch_keys",
    "combine_pair_spectra",
    "detector_noise",
    "network_spectra_ensemble",
    "population_key",
    "realization_key",
    "simulate_ensemble",
    "simulate_realization",
    "union_configuration",
]
