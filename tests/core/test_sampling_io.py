"""Tests for spectrum-only HDF5 I/O."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from astrogwb.sampling._io import FORMAT_NAME, SpectraArtifact, load_spectra, save_spectra
from astrogwb.waveform import PolarizationPowerGenerator


def _artifact() -> SpectraArtifact:
    waveform = PolarizationPowerGenerator(
        approximant="TaylorF2",
        minimum_frequency=20.0,
        maximum_frequency=32.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
        frequency_resolution=4.0,
    )
    return SpectraArtifact(
        frequencies=np.array([20.0, 24.0, 28.0, 32.0]),
        spectral_density=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ]
        ),
        n_events=np.array([11, 13]),
        total_merger_rate=np.array([1.5, 1.5]),
        hyperparameters={"H0": np.array([67.0, 67.0]), "local_merger_rate": np.array([800.0, 800.0])},
        source_model_name="bns_md_cosmological",
        rate_model_name="madau_dickinson",
        model_kwargs={"z_min": 0.1, "z_max": 10.0, "n_grid": 32},
        density_sites=("redshift", "source_frame_mass_1", "source_frame_mass_2"),
        waveform_metadata=waveform,
        n_max_sigma=5.0,
        seed=7,
    )


def test_round_trip_preserves_spectra_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "spectra.h5"
    expected = _artifact()
    save_spectra(expected, path)

    with h5py.File(path) as handle:
        assert handle.attrs["format_name"] == FORMAT_NAME
        assert set(handle) == {
            "frequency",
            "spectral_density",
            "n_events",
            "total_merger_rate",
            "hyperparameters",
        }
        assert json.loads(handle.attrs["population_model_kwargs"]) == {
            "n_grid": 32,
            "z_max": 10.0,
            "z_min": 0.1,
        }

    actual = load_spectra(path)
    np.testing.assert_array_equal(actual.frequencies, expected.frequencies)
    np.testing.assert_array_equal(actual.spectral_density, expected.spectral_density)
    np.testing.assert_array_equal(actual.n_events, expected.n_events)
    np.testing.assert_array_equal(actual.total_merger_rate, expected.total_merger_rate)
    assert actual.source_model_name == expected.source_model_name
    assert actual.rate_model_name == expected.rate_model_name
    assert actual.model_kwargs == expected.model_kwargs
    assert actual.density_sites == expected.density_sites
    assert actual.n_max_sigma == expected.n_max_sigma
    assert actual.seed == expected.seed
    for name in expected.hyperparameters:
        np.testing.assert_array_equal(actual.hyperparameters[name], expected.hyperparameters[name])


@pytest.mark.parametrize("dataset", ["frequency", "spectral_density", "n_events", "total_merger_rate", "hyperparameters"])
def test_missing_dataset_is_rejected(tmp_path: Path, dataset: str) -> None:
    path = tmp_path / "invalid.h5"
    save_spectra(_artifact(), path)
    with h5py.File(path, "r+") as handle:
        del handle[dataset]
        with pytest.raises(ValueError, match="missing"):
            load_spectra(path)
