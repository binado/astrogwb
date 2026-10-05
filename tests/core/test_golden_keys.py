"""Literal content keys and node paths: nothing may re-key a cache by accident.

Each record is built from literal fields with ``version`` pinned, so only a
change to the canonical payload (or its hash) can move a key. The literals were
captured when the population record became composed sub-models; a
deliberate change to a record or to the path scheme re-captures them.
"""

from __future__ import annotations

import numpy as np

from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import ComponentMetadata, PopulationMetadata
from astrogwb.simulators.core import split_seed
from astrogwb.simulators.polarization_power import CatalogMetadata, polarization_power
from astrogwb.simulators.spectra import SpectraMetadata, spectra
from astrogwb.waveform import WaveformMetadata

CATALOG_KEY = "a4f77eea8d97ff0c"
SPECTRA_KEY = "f461dfccb0a84990"
CATALOG_PATH = "polarization_power-a4f77eea8d97ff0c-a6d964d9e7c90939.h5"
SPECTRA_PATH = "spectra-f461dfccb0a84990-ccfa6eba0950be8d.h5"
SPLIT_SEEDS = [15502207689350057789, 3547686303310379753, 4445048811325225245]


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
        model_name="bns_madau_dickinson",
        model_kwargs={"sample_inclination": True},
        redshift=ComponentMetadata(
            model="madau_dickinson",
            kwargs={"minimum_redshift": 0.0, "maximum_redshift": 2, "n_grid": 32},
        ),
        mass=ComponentMetadata(model="ordered_uniform"),
    )


def _catalog_metadata() -> CatalogMetadata:
    return CatalogMetadata(
        waveform=_waveform(),
        population=_population(),
        fiducials={"H0": 67.9, "Om0": 0.31},
        num_samples=100,
        version="0.3.0",
    )


def test_catalog_metadata_key_is_stable() -> None:
    assert _catalog_metadata().key() == CATALOG_KEY


def test_polarization_power_path_is_stable() -> None:
    path = polarization_power.path(
        {"seed": np.uint64(41)}, _catalog_metadata(), "cache"
    )
    assert path.name == CATALOG_PATH


def test_split_seed_is_stable_and_prefix_stable() -> None:
    np.testing.assert_array_equal(
        split_seed(41, 3), np.array(SPLIT_SEEDS, dtype=np.uint64)
    )
    np.testing.assert_array_equal(split_seed(41, 2), split_seed(41, 3)[:2])


def _spectra_metadata() -> SpectraMetadata:
    return SpectraMetadata(
        waveform=_waveform(),
        population=_population(),
        hyperparameters={
            "H0": 67.9,
            "Om0": DistributionConfig(dist="Uniform", kwargs={"low": 0.2, "high": 0.4}),
        },
        observation_time=1.0,
        count="fixed",
        num_events=10,
        version="0.3.0",
    )


def test_spectra_metadata_key_is_stable() -> None:
    assert _spectra_metadata().key() == SPECTRA_KEY


def test_spectra_path_is_stable() -> None:
    path = spectra.path({"seeds": split_seed(41, 3)}, _spectra_metadata(), "cache")
    assert path.name == SPECTRA_PATH
