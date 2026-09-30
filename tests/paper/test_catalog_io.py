"""Tests for the direct HDF5 catalog format."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from catalog_fixtures import make_catalog

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.catalog._io import CATALOG_FORMAT_NAME, validate_catalog_file
from astrogwb.metadata import CatalogMetadata


def test_hdf5_layout_metadata_and_order_round_trip(tmp_path: Path) -> None:
    catalog = make_catalog(redshift=np.array([0.1, 0.5, 1.0]))
    path = tmp_path / "catalog.h5"
    catalog.save(path)
    with h5py.File(path) as handle:
        assert set(handle) == {"frequency", "polarization_power", "source_parameters"}
        assert handle.attrs["format_name"] == CATALOG_FORMAT_NAME
        assert set(handle.attrs) == {
            "format_name",
            "domain",
            "metadata",
            "source_parameter_names",
        }
        assert handle.attrs["metadata"] == catalog.metadata.model_dump_json()
        assert (
            CatalogMetadata.model_validate_json(handle.attrs["metadata"])
            == catalog.metadata
        )
        assert json.loads(handle.attrs["source_parameter_names"]) == list(
            catalog.source_parameters
        )
        assert handle["source_parameters"].dtype == np.float64
    restored = PolarizationPowerCatalog.load(path)
    assert list(restored.source_parameters) == list(catalog.source_parameters)
    np.testing.assert_array_equal(restored.frequencies, catalog.frequencies)
    np.testing.assert_array_equal(
        restored.polarization_power, catalog.polarization_power
    )
    np.testing.assert_allclose(restored.bin_widths, catalog.bin_widths)
    assert restored.metadata == catalog.metadata
    assert restored.metadata.key() == catalog.metadata.key()
    for name in catalog.source_parameters:
        np.testing.assert_array_equal(
            restored.source_parameters[name], catalog.source_parameters[name]
        )


def test_compression_applies_to_arrays(tmp_path: Path) -> None:
    path = tmp_path / "compressed.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path, compression="gzip")
    with h5py.File(path) as handle:
        assert all(handle[name].compression == "gzip" for name in handle)


@pytest.mark.parametrize(
    "dataset", ["frequency", "polarization_power", "source_parameters"]
)
def test_missing_dataset_is_rejected(tmp_path: Path, dataset: str) -> None:
    path = tmp_path / "invalid.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        del handle[dataset]
        with pytest.raises(ValueError, match="missing"):
            validate_catalog_file(handle, label="test")


def test_unknown_format_and_domain_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["format_name"] = "foreign"
        with pytest.raises(ValueError, match="format_name"):
            validate_catalog_file(handle, label="test")
        handle.attrs["format_name"] = CATALOG_FORMAT_NAME
        handle.attrs["domain"] = "time"
        with pytest.raises(ValueError, match="domain"):
            validate_catalog_file(handle, label="test")


def test_malformed_source_shape_and_dtype_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        data = np.asarray(handle["source_parameters"])
        del handle["source_parameters"]
        handle.create_dataset("source_parameters", data=data.astype(np.float32))
        with pytest.raises(ValueError, match="float64"):
            validate_catalog_file(handle, label="test")


def test_invalid_metadata_and_unknown_population_fail_on_load(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        metadata = json.loads(handle.attrs["metadata"])
        metadata["population"]["model_name"] = "no_such_population"
        handle.attrs["metadata"] = json.dumps(metadata)
    with pytest.raises(KeyError):
        PolarizationPowerCatalog.load(path)


def test_missing_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        del handle.attrs["metadata"]
    with pytest.raises(ValueError, match="invalid.h5: missing.*metadata"):
        PolarizationPowerCatalog.load(path)


@pytest.mark.parametrize("legacy", ["v5", "v8"])
def test_legacy_format_is_rejected(tmp_path: Path, legacy: str) -> None:
    path = tmp_path / "legacy.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["format_name"] = f"astrogwb_catalog_{legacy}"
    with pytest.raises(ValueError, match="format_name"):
        PolarizationPowerCatalog.load(path)


@pytest.mark.parametrize("payload", ["{not json", "[]", "null", 42])
def test_malformed_metadata_is_rejected(tmp_path: Path, payload: str | int) -> None:
    path = tmp_path / "invalid.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["metadata"] = payload
    with pytest.raises(ValueError, match="invalid.h5: invalid catalog metadata"):
        PolarizationPowerCatalog.load(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_samples", "4"), ("num_samples", True), ("unknown", 1), ("waveform", {})],
)
def test_invalid_json_metadata_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "invalid.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        metadata = json.loads(handle.attrs["metadata"])
        metadata[field] = value
        handle.attrs["metadata"] = json.dumps(metadata)
    with pytest.raises(
        ValueError, match=f"(?s)invalid.h5: invalid catalog metadata:.*{field}"
    ):
        PolarizationPowerCatalog.load(path)


def test_stored_sample_count_is_checked_against_arrays(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        metadata = json.loads(handle.attrs["metadata"])
        metadata["num_samples"] = 3
        handle.attrs["metadata"] = json.dumps(metadata)
    with pytest.raises(
        ValueError, match="invalid.h5: metadata num_samples.*sample dimension"
    ):
        PolarizationPowerCatalog.load(path)
