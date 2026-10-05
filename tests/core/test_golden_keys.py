"""Literal content keys: a move of the metadata classes must not re-key a cache.

Each record is built from literal fields with ``version`` pinned, so only a
change to the canonical payload (or its hash) can move the key. The literals
below were captured before the ``astrogwb.simulators`` layout move.
"""

from __future__ import annotations

from astrogwb.metadata import PriorSpec
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.polarization_power import CatalogMetadata
from astrogwb.simulators.spectra import SpectraMetadata
from astrogwb.waveform import WaveformMetadata

CATALOG_KEY = "c29b8afcba2c3726"
SPECTRA_KEY = "de4154fe38da85a8"


def _waveform() -> WaveformMetadata:
    return WaveformMetadata(
        approximant="IMRPhenomD_NRTidalv2",
        minimum_frequency=10.0,
        maximum_frequency=512.0,
        reference_frequency=20.0,
        sampling_frequency=2048.0,
        frequency_resolution=0.25,
    )


def _population() -> PopulationMetadata:
    return PopulationMetadata(
        model_name="bns_md_cosmological",
        model_kwargs={
            "minimum_redshift": 0.0,
            "maximum_redshift": 2,
            "n_grid": 32,
            "sample_inclination": True,
        },
        seed=41,
    )


def test_catalog_metadata_key_is_stable() -> None:
    metadata = CatalogMetadata(
        waveform=_waveform(),
        population=_population(),
        fiducials={"H0": 67.9, "Om0": 0.31},
        num_samples=100,
        version="0.3.0",
    )
    assert metadata.key() == CATALOG_KEY


def test_spectra_metadata_key_is_stable() -> None:
    metadata = SpectraMetadata(
        waveform=_waveform(),
        population=_population(),
        hyperparameters={
            "H0": 67.9,
            "Om0": PriorSpec(dist="Uniform", kwargs={"low": 0.2, "high": 0.4}),
        },
        num_draws=4,
        observation_time=1.0,
        count="fixed",
        num_events=10,
        version="0.3.0",
    )
    assert metadata.key() == SPECTRA_KEY
