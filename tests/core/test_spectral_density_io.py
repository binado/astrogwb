"""Tests for the spectral-density HDF5 format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from astrogwb.catalog import SpectralDensityCatalog
from astrogwb.catalog._io import (
    REQUIRED_SPECTRAL_DENSITY_ATTRS,
    SPECTRAL_DENSITY_DATASETS,
    SPECTRAL_DENSITY_FORMAT_NAME,
)
from astrogwb.populations import PopulationRecord
from astrogwb.waveform import PolarizationPowerGenerator


def _catalog(**overrides: Any) -> SpectralDensityCatalog:
    waveform = PolarizationPowerGenerator(
        approximant="TaylorF2",
        minimum_frequency=20.0,
        maximum_frequency=32.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
        frequency_resolution=4.0,
    )
    fields: dict[str, Any] = {
        "frequencies": np.array([20.0, 24.0, 28.0, 32.0]),
        "spectral_density": np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ]
        ),
        "n_events": np.array([11, 13]),
        "total_merger_rate": np.array([1.5, 1.5]),
        "hyperparameters": {
            "H0": np.array([67.0, 67.0]),
            "local_merger_rate": np.array([800.0, 800.0]),
        },
        "waveform_metadata": waveform,
        "_population": PopulationRecord(
            source_model_name="bns_md_cosmological",
            rate_model_name="madau_dickinson",
            model_kwargs={"z_min": 0.1, "z_max": 10.0, "n_grid": 32},
            density_sites=("redshift", "source_frame_mass_1", "source_frame_mass_2"),
            seed=7,
        ),
        "n_max_sigma": 5.0,
        "observation_time": 1.0,
    }
    return SpectralDensityCatalog(**{**fields, **overrides})


def test_round_trip_preserves_spectra_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "spectra.h5"
    expected = _catalog()
    expected.save(path)

    with h5py.File(path) as handle:
        assert handle.attrs["format_name"] == SPECTRAL_DENSITY_FORMAT_NAME
        assert set(handle) == set(SPECTRAL_DENSITY_DATASETS)
        assert json.loads(handle.attrs["population_model_kwargs"]) == {
            "n_grid": 32,
            "z_max": 10.0,
            "z_min": 0.1,
        }
        assert json.loads(handle.attrs["source_parameter_names"]) == [
            "H0",
            "local_merger_rate",
        ]
        assert "average_mode" not in handle.attrs
        assert handle.attrs["observation_time"] == 1.0

    actual = SpectralDensityCatalog.load(path)
    np.testing.assert_array_equal(actual.frequencies, expected.frequencies)
    np.testing.assert_array_equal(actual.spectral_density, expected.spectral_density)
    np.testing.assert_array_equal(actual.n_events, expected.n_events)
    np.testing.assert_array_equal(actual.total_merger_rate, expected.total_merger_rate)
    assert actual.population == expected.population
    assert actual.n_max_sigma == expected.n_max_sigma
    assert actual.seed == expected.seed
    assert actual.observation_time == expected.observation_time
    for name in expected.hyperparameters:
        np.testing.assert_array_equal(
            actual.hyperparameters[name], expected.hyperparameters[name]
        )


def test_compression_applies_to_arrays(tmp_path: Path) -> None:
    path = tmp_path / "compressed.h5"
    _catalog().save(path, compression="gzip")
    with h5py.File(path) as handle:
        assert all(handle[name].compression == "gzip" for name in handle)


@pytest.mark.parametrize("dataset", SPECTRAL_DENSITY_DATASETS)
def test_missing_dataset_is_rejected(tmp_path: Path, dataset: str) -> None:
    path = tmp_path / "invalid.h5"
    _catalog().save(path)
    with h5py.File(path, "r+") as handle:
        del handle[dataset]
    with pytest.raises(ValueError, match="missing"):
        SpectralDensityCatalog.load(path)


@pytest.mark.parametrize("attr", REQUIRED_SPECTRAL_DENSITY_ATTRS)
def test_missing_population_attribute_is_rejected(tmp_path: Path, attr: str) -> None:
    path = tmp_path / "invalid.h5"
    _catalog().save(path)
    with h5py.File(path, "r+") as handle:
        del handle.attrs[attr]
    with pytest.raises(ValueError, match=attr):
        SpectralDensityCatalog.load(path)


def test_unknown_format_and_domain_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    _catalog().save(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["format_name"] = "foreign"
    with pytest.raises(ValueError, match="format_name"):
        SpectralDensityCatalog.load(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["format_name"] = SPECTRAL_DENSITY_FORMAT_NAME
        handle.attrs["domain"] = "time"
    with pytest.raises(ValueError, match="domain"):
        SpectralDensityCatalog.load(path)


def test_legacy_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.h5"
    _catalog().save(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["format_name"] = "astrogwb_spectral_density_v1"
    with pytest.raises(ValueError, match="format_name"):
        SpectralDensityCatalog.load(path)


def test_hyperparameters_must_be_serialized_as_float64(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    _catalog().save(path)
    with h5py.File(path, "r+") as handle:
        values = np.asarray(handle["hyperparameters"])
        del handle["hyperparameters"]
        handle.create_dataset("hyperparameters", data=values.astype(np.float32))
    with pytest.raises(ValueError, match="float64"):
        SpectralDensityCatalog.load(path)


def test_unknown_population_is_rejected_on_load(tmp_path: Path) -> None:
    path = tmp_path / "unknown.h5"
    _catalog().save(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["population_source_model"] = "not_registered"
    with pytest.raises(KeyError, match="not_registered"):
        SpectralDensityCatalog.load(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("spectral_density", np.zeros((2, 3)), "spectral_density"),
        ("n_events", np.array([1, 2, 3]), "n_events"),
        ("total_merger_rate", np.array([1.0]), "total_merger_rate"),
        ("frequencies", np.zeros((2, 2)), "one-dimensional"),
        ("observation_time", 0.0, "observation_time"),
        ("n_max_sigma", -1.0, "n_max_sigma"),
    ],
)
def test_malformed_fields_are_rejected_at_construction(
    field: str, value: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _catalog(**{field: value})


def test_hyperparameter_column_length_must_match_draws() -> None:
    with pytest.raises(ValueError, match="H0"):
        _catalog(hyperparameters={"H0": np.array([67.0, 67.0, 67.0])})
