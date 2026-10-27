"""The tiny closed-form setup the simulator tests share.

A BNS population of a few tens of sources at ``observation_time=1e-4`` years,
the closed-form inspiral on a four-bin grid and a three-detector network: every
stage runs for real in well under a second.
"""

from __future__ import annotations

from astrogwb.populations import PopulationMetadata as PopulationSpec
from astrogwb.simulators import DetectorConfiguration, PopulationMetadata
from astrogwb.waveform import WaveformMetadata

WAVEFORM = WaveformMetadata(
    approximant="AnalyticInspiral",
    minimum_frequency=20.0,
    maximum_frequency=50.0,
    reference_frequency=20.0,
    sampling_frequency=128.0,
    frequency_resolution=10.0,
)
DETECTORS = DetectorConfiguration(
    detectors=["S1", "R1", "C1"],
    psd_references={
        "S1": "ET_COBA_15km_psd.txt",
        "R1": "ET_COBA_15km_psd.txt",
        "C1": "cosmic_explorer_strain_psd.txt",
    },
)


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
