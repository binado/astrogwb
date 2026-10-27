"""The tiny closed-form setup the simulator tests share.

A BNS population of a few tens of sources at ``observation_time=1e-4`` years,
the closed-form inspiral on a four-bin grid and two networks, a three-detector
one and a pair it contains: every stage runs for real in well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astrogwb.detector.config import load_detector_config
from astrogwb.populations import PopulationMetadata as PopulationSpec
from astrogwb.simulators import (
    BackgroundArtifact,
    DetectorRegistry,
    PopulationMetadata,
)
from astrogwb.waveform import WaveformMetadata

WAVEFORM = WaveformMetadata(
    approximant="AnalyticInspiral",
    minimum_frequency=20.0,
    maximum_frequency=50.0,
    reference_frequency=20.0,
    sampling_frequency=128.0,
    frequency_resolution=10.0,
)
_root = Path(__file__).parents[2]
_base = load_detector_config([_root / "config/detectors.toml"])
NETWORKS = DetectorRegistry(
    detectors=_base.detectors,
    networks={"triangle": ("S1", "R1", "C1"), "pair": ("S1", "C1")},
)
DETECTORS = NETWORKS


def assert_same_spectra(
    actual: BackgroundArtifact, expected: BackgroundArtifact
) -> None:
    """Every network's pair and network spectra are equal, bit for bit."""
    assert actual.spectra.keys() == expected.spectra.keys()
    for name, spectrum in expected.spectra.items():
        for field, values in spectrum._asdict().items():
            np.testing.assert_array_equal(getattr(actual.spectra[name], field), values)


def population_metadata(observation_time: float = 1e-4) -> PopulationMetadata:
    """The BNS population observed for ``observation_time`` years."""
    return PopulationMetadata(
        population=PopulationSpec(
            model_name="bns_coba",
            model_kwargs={
                "mass_model": "uniform",
                "minimum_redshift": 0.0,
                "maximum_redshift": 5.0,
                "n_grid": 64,
            },
        ),
        hyperparameters={
            "H0": 67.66,
            "Omega_m": 0.3096,
            "gamma": 1.42,
            "kappa": 4.62,
            "z_peak": 1.84,
            "minimum_mass": 1.0,
            "mass_width": 1.5,
            "local_merger_rate": 750.0,
        },
        observation_time=observation_time,
    )
