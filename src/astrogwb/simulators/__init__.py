"""Poisson realizations of the background, in three stages.

1. :class:`PopulationSimulator` draws one realization -- hyperparameters, a
   Poisson count, every source's intrinsic and extrinsic parameters -- as a
   :class:`PopulationArtifact`.
2. :class:`WaveformSimulator` turns it into complex radiation-frame
   polarizations, :class:`Waveforms`, generated chunk by chunk; written, they
   are a :class:`WaveformArtifact`.
3. :class:`BackgroundSimulator` projects them onto a
   :class:`DetectorConfiguration` and sums the per-pair cross-powers into a
   :class:`BackgroundArtifact`; :func:`combine_pair_spectra` weights the pairs
   into one network spectrum.

Each stage runs alone, writing ``metadata.toml`` and ``data.h5`` that the
next stage opens, or fused in memory: ``BackgroundSimulator(...)(
WaveformSimulator(...)(population))`` never holds more than one chunk of
polarizations. Only the population stage draws random numbers.
"""

from astrogwb.simulators._io import batch_keys
from astrogwb.simulators.background import (
    BackgroundArtifact,
    BackgroundSimulator,
    DetectorConfiguration,
    combine_pair_spectra,
)
from astrogwb.simulators.population import (
    PopulationArtifact,
    PopulationMetadata,
    PopulationSimulator,
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
    "PopulationArtifact",
    "PopulationMetadata",
    "PopulationSimulator",
    "WaveformArtifact",
    "WaveformSimulator",
    "Waveforms",
    "batch_keys",
    "combine_pair_spectra",
]
