"""Round-trip and format tests for the paper-owned xarray catalog I/O."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from astrogwb.catalog import Catalog, PopulationMetadata
from astrogwb.catalog.io import (
    DOMAIN_FREQUENCY,
    FORMAT_NAME,
    RESERVED_ATTRS,
    catalog_from_dataset,
    catalog_to_dataset,
    load_catalog,
    open_catalog,
    save_catalog,
    validate_catalog_dataset,
)
from astrogwb.waveform import PolarizationPowerGenerator


def _catalog(
    *,
    provenance: dict[str, str | int | float] | None = None,
    source_type: str | None = "bns",
) -> Catalog:
    frequencies = np.array([10.0, 20.0, 30.0, 40.0])
    return Catalog(
        source_parameters={
            "redshift": np.array([0.1, 0.5, 1.0]),
            "integer_parameter": np.array([1, 2, 3], dtype=np.int16),
        },
        polarization_power=np.arange(12, dtype=np.float64).reshape(4, 3),
        waveform_metadata=PolarizationPowerGenerator(
            frequencies=frequencies,
            approximant="Toy",
            minimum_frequency=10.0,
            maximum_frequency=40.0,
            reference_frequency=20.0,
            sampling_frequency=128.0,
            df=10.0,
        ),
        population_metadata=PopulationMetadata(
            name="madau-dickinson",
            seed=41,
            num_samples=3,
            source_type=source_type,
            provenance={
                "producer": "test",
                "version": 2,
                "threshold": 0.25,
                **({} if provenance is None else provenance),
            },
        ),
    )


def test_catalog_to_dataset_uses_stacked_float64_format() -> None:
    dataset = catalog_to_dataset(_catalog())

    assert dataset.polarization_power.dims == ("frequency", "sample")
    assert dataset.source_parameters.dims == ("sample", "parameter")
    assert dataset.source_parameters.dtype == np.float64
    assert dataset.parameter.values.tolist() == ["redshift", "integer_parameter"]
    assert dataset.attrs == {
        "format_name": FORMAT_NAME,
        "domain": DOMAIN_FREQUENCY,
        "approximant": "Toy",
        "minimum_frequency": 10.0,
        "maximum_frequency": 40.0,
        "reference_frequency": 20.0,
        "sampling_frequency": 128.0,
        "df": 10.0,
        "population_name": "madau-dickinson",
        "population_seed": 41,
        "population_num_samples": 3,
        "population_source_type": "bns",
        "producer": "test",
        "version": 2,
        "threshold": 0.25,
    }


def test_catalog_dataset_file_dataset_catalog_round_trip(tmp_path: Path) -> None:
    original = _catalog()
    path = tmp_path / "catalog.h5"

    initial_dataset = catalog_to_dataset(original)
    save_catalog(path, catalog_from_dataset(initial_dataset))
    loaded_dataset = load_catalog(path)
    restored = catalog_from_dataset(loaded_dataset)

    assert type(restored.waveform_metadata) is PolarizationPowerGenerator
    np.testing.assert_array_equal(
        restored.waveform_metadata.frequencies,
        original.waveform_metadata.frequencies,
    )
    np.testing.assert_array_equal(
        restored.polarization_power, original.polarization_power
    )
    for name, values in original.source_parameters.items():
        np.testing.assert_allclose(restored.source_parameters[name], values)
        assert restored.source_parameters[name].dtype == np.float64
    assert restored.population_metadata.name == "madau-dickinson"
    assert restored.population_metadata.seed == 41
    assert restored.population_metadata.num_samples == 3
    assert restored.population_metadata.source_type == "bns"
    assert restored.population_metadata.provenance == {
        "producer": "test",
        "version": 2,
        "threshold": 0.25,
    }


def test_save_with_compression_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "catalog.h5"
    save_catalog(path, _catalog(), compression="gzip")

    with h5py.File(path) as handle:
        assert handle["polarization_power"].compression == "gzip"
    np.testing.assert_array_equal(
        load_catalog(path).polarization_power.values,
        _catalog().polarization_power,
    )


def test_open_catalog_leaves_polarization_power_lazy(tmp_path: Path) -> None:
    path = tmp_path / "catalog.h5"
    save_catalog(path, _catalog())

    with open_catalog(path) as dataset:
        assert dataset.polarization_power._in_memory is False


@pytest.mark.parametrize("reserved", sorted(RESERVED_ATTRS))
def test_provenance_rejects_reserved_names(reserved: str) -> None:
    with pytest.raises(ValueError, match="reserved catalog attribute"):
        catalog_to_dataset(_catalog(provenance={reserved: "hijacked"}))


def test_optional_source_type_is_omitted_and_decodes_as_none() -> None:
    dataset = catalog_to_dataset(_catalog(source_type=None))

    assert "population_source_type" not in dataset.attrs
    assert catalog_from_dataset(dataset).population_metadata.source_type is None


def test_old_waveform_catalog_format_requires_regeneration(tmp_path: Path) -> None:
    path = tmp_path / "old.h5"
    dataset = catalog_to_dataset(_catalog())
    dataset.attrs["format_name"] = "waveform_catalog"
    dataset.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="obsolete.*regenerate"):
        load_catalog(path)


def test_unknown_format_and_domain_are_rejected() -> None:
    dataset = catalog_to_dataset(_catalog())

    with pytest.raises(ValueError, match="format_name"):
        validate_catalog_dataset(
            dataset.assign_attrs(format_name="foreign"), label="test"
        )
    with pytest.raises(ValueError, match="domain"):
        validate_catalog_dataset(dataset.assign_attrs(domain="time"), label="test")


def test_dataset_validation_rejects_malformed_layout_and_sample_metadata() -> None:
    dataset = catalog_to_dataset(_catalog())

    wrong_dims = dataset.rename_dims({"sample": "event"})
    with pytest.raises(ValueError, match="frequency, sample"):
        validate_catalog_dataset(wrong_dims, label="test")
    with pytest.raises(ValueError, match="population_num_samples"):
        validate_catalog_dataset(
            dataset.assign_attrs(population_num_samples=4), label="test"
        )


def test_dataset_validation_does_not_require_loading_power(tmp_path: Path) -> None:
    path = tmp_path / "catalog.h5"
    save_catalog(path, _catalog())

    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        validate_catalog_dataset(dataset, label="test")
        assert dataset.polarization_power._in_memory is False
