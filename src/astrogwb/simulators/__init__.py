"""Poisson realizations of the background, simulated in one pass.

A :class:`SimulationConfig` holds everything that determines a run: the
population, the waveform, the detector registry and an optional noise segment
length. :func:`simulate` draws it at a key in one pass, with two building
blocks:

1. :class:`PopulationSimulator` draws one realization -- hyperparameters, a
   Poisson count, every source's intrinsic and extrinsic parameters -- as a
   :class:`PopulationArtifact`.
2. :class:`BackgroundSimulator` generates each chunk's radiation-frame
   polarizations through a :class:`~astrogwb.waveform.WaveformMetadata`,
   projects them onto the named networks of a
   :class:`~astrogwb.detector.DetectorRegistry`, and sums the per-pair
   cross-powers into a :class:`BackgroundArtifact` holding each network's pair
   spectra; :func:`combine_pair_spectra` weights them into a network spectrum.
   With a ``segment_duration`` it assigns sources to segments and
   cross-correlates each segment's coherent strain plus fresh detector noise;
   :meth:`BackgroundArtifact.noise` derives the networks' likelihood inputs, a
   :class:`NetworkNoise` each.

Only the pair spectra are kept. A :class:`BackgroundArtifact` writes one HDF5
file carrying its config as JSON and the package version;
:func:`network_spectra_ensemble` stacks an ensemble of such files per network.
Only the population stage draws random numbers; the background-stage noise is a
deterministic function of the realization's key.
"""

from astrogwb.simulators._io import realization_key
from astrogwb.simulators.background import (
    BackgroundArtifact,
    BackgroundSimulator,
    NetworkNoise,
    combine_pair_spectra,
)
from astrogwb.simulators.config import SimulationConfig
from astrogwb.simulators.ensemble import NetworkSpectra, network_spectra_ensemble
from astrogwb.simulators.population import (
    PopulationArtifact,
    PopulationMetadata,
    PopulationSimulator,
)
from astrogwb.simulators.simulation import simulate

__all__ = [
    "BackgroundArtifact",
    "BackgroundSimulator",
    "NetworkNoise",
    "NetworkSpectra",
    "PopulationArtifact",
    "PopulationMetadata",
    "PopulationSimulator",
    "SimulationConfig",
    "combine_pair_spectra",
    "network_spectra_ensemble",
    "realization_key",
    "simulate",
]
