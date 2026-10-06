"""Literal content keys, file stems and batched keys: nothing may re-key a file by accident.

Each record is built from literal fields with ``version`` pinned, so only a
change to the canonical payload (or its hash) can move a key. The literals were
captured when seeds became simulator inputs rather than metadata fields; a
deliberate change to a record or to the path scheme re-captures them.
"""

from __future__ import annotations

import jax
import numpy as np

from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.core import batch_keys
from astrogwb.simulators.polarization_power import CatalogMetadata, catalog_stem
from astrogwb.simulators.spectra import SpectraMetadata
from astrogwb.waveform import WaveformMetadata

CATALOG_KEY = "1a7d74705a3804d5"
SPECTRA_KEY = "ae70dadcc56cadd5"
POPULATION_KEY = "72e6e431f2316e25"
CATALOG_STEM = f"polarization_power-{CATALOG_KEY}-41"


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


def test_catalog_stem_is_the_key_and_the_seed() -> None:
    assert catalog_stem(_catalog_metadata(), np.uint64(41)) == CATALOG_STEM


def test_batch_keys_are_prefix_stable_and_seed_dependent() -> None:
    keys = jax.random.key_data(batch_keys(41, 3))
    np.testing.assert_array_equal(jax.random.key_data(batch_keys(41, 2)), keys[:2])
    assert not np.array_equal(jax.random.key_data(batch_keys(42, 3)), keys)
    assert len({tuple(row) for row in np.asarray(keys).tolist()}) == 3


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


def test_population_key_is_the_waveform_free_part_of_a_spectra_record() -> None:
    assert _spectra_metadata().sources.key() == POPULATION_KEY
