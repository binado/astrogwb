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


def test_hdf5_layout_metadata_and_order_round_trip(tmp_path: Path) -> None:
    catalog = make_catalog(redshift=np.array([0.1, 0.5, 1.0]))
    path = tmp_path / "catalog.h5"
    catalog.save(path)
    with h5py.File(path) as handle:
        assert set(handle) == {"frequency", "polarization_power", "source_parameters"}
        assert handle.attrs["format_name"] == CATALOG_FORMAT_NAME
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
    assert restored.df == catalog.df


def test_a_file_carrying_the_retired_density_sites_attribute_still_loads(
    tmp_path: Path,
) -> None:
    """Catalogs generated before the weighting choice moved need no regeneration.

    Every ``.h5`` written while ``density_sites`` was part of the population
    record carries a ``population_density_sites`` attribute. The reader names
    the attributes it wants, so the leftover is not read -- and the eight
    committed catalogs are GPU jobs, which is why this is a compatibility
    guarantee rather than a rebuild.
    """
    path = tmp_path / "pre-migration.h5"
    catalog = make_catalog(redshift=np.linspace(0.1, 1.0, 4))
    catalog.save(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["population_density_sites"] = json.dumps(
            ["redshift", "source_frame_mass_1", "source_frame_mass_2"]
        )

    restored = PolarizationPowerCatalog.load(path)
    assert restored.population == catalog.population
    np.testing.assert_array_equal(
        restored.polarization_power, catalog.polarization_power
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
        handle.attrs["population_model"] = "no_such_population"
    with pytest.raises(KeyError):
        PolarizationPowerCatalog.load(path)


def test_missing_population_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        del handle.attrs["population_model"]
    with pytest.raises(ValueError, match="population_model"):
        PolarizationPowerCatalog.load(path)


def test_legacy_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.h5"
    make_catalog(redshift=np.linspace(0.1, 1.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["format_name"] = "astrogwb_catalog_v5"
    with pytest.raises(ValueError, match="format_name"):
        PolarizationPowerCatalog.load(path)
