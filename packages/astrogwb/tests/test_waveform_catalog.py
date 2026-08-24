from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from astrogwb.waveform.catalog import (
    DOMAIN_FREQUENCY,
    FORMAT_NAME,
    FORMAT_VERSION,
    load_catalog,
    make_catalog,
    open_catalog,
    save_catalog,
    validate_catalog,
)


def _catalog(nsamples: int = 6, nfreq: int = 4):
    rng = np.random.default_rng(0)
    frequencies = np.linspace(10.0, 40.0, nfreq)
    plus = rng.normal(size=(nsamples, nfreq)) + 1j * rng.normal(size=(nsamples, nfreq))
    cross = rng.normal(size=(nsamples, nfreq)) + 1j * rng.normal(size=(nsamples, nfreq))
    source_parameters = {
        "redshift": rng.uniform(0.0, 2.0, nsamples),
        "luminosity_distance": rng.uniform(100.0, 1000.0, nsamples),
    }
    return make_catalog(
        frequencies=frequencies,
        plus=plus,
        cross=cross,
        source_parameters=source_parameters,
        approximant="Toy",
        minimum_frequency=float(frequencies[0]),
        maximum_frequency=float(frequencies[-1]),
        reference_frequency=20.0,
        sampling_frequency=128.0,
    )


def test_round_trip_preserves_values_and_attrs(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.h5"

    save_catalog(path, catalog)
    loaded = load_catalog(path)

    np.testing.assert_array_equal(loaded.frequency.values, catalog.frequency.values)
    np.testing.assert_allclose(
        loaded.polarizations.values, catalog.polarizations.values
    )
    assert loaded.polarizations.dtype == np.complex128
    np.testing.assert_allclose(
        loaded.source_parameters.values, catalog.source_parameters.values
    )
    assert dict(loaded.attrs) == {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "domain": DOMAIN_FREQUENCY,
        "approximant": "Toy",
        "minimum_frequency": 10.0,
        "maximum_frequency": 40.0,
        "reference_frequency": 20.0,
        "sampling_frequency": 128.0,
    }


def test_save_with_compression_round_trips(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.h5"

    save_catalog(path, catalog, compression="gzip")

    with h5py.File(path) as f:
        assert f["polarizations"].compression == "gzip"
    loaded = load_catalog(path)
    np.testing.assert_allclose(
        loaded.polarizations.values, catalog.polarizations.values
    )


def test_open_catalog_leaves_polarizations_lazy(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.h5"
    save_catalog(path, catalog)

    opened = open_catalog(path)

    assert opened.polarizations._in_memory is False


def test_load_catalog_rejects_wrong_format_name(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.h5"
    save_catalog(path, catalog)
    with h5py.File(path, "r+") as f:
        f.attrs["format_name"] = "something_else"

    with pytest.raises(ValueError, match="format_name"):
        load_catalog(path)


def test_load_catalog_rejects_wrong_format_version(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.h5"
    save_catalog(path, catalog)
    with h5py.File(path, "r+") as f:
        f.attrs["format_version"] = np.int64(1)

    with pytest.raises(
        ValueError, match="v1 \\(pluscross\\) catalogs must be regenerated"
    ):
        load_catalog(path)


def test_load_catalog_rejects_wrong_domain(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.h5"
    save_catalog(path, catalog)
    with h5py.File(path, "r+") as f:
        f.attrs["domain"] = "time"

    with pytest.raises(ValueError, match="domain"):
        load_catalog(path)


def test_validate_catalog_rejects_non_monotonic_frequencies() -> None:
    catalog = _catalog()
    bad = catalog.assign_coords(frequency=catalog.frequency.values[::-1])

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_catalog(bad, label="test")


def test_validate_catalog_rejects_missing_polarizations() -> None:
    catalog = _catalog()
    bad = catalog.drop_vars("polarizations")

    with pytest.raises(ValueError, match="polarizations"):
        validate_catalog(bad, label="test")


def test_validate_catalog_rejects_wrong_polarization_axis_size() -> None:
    catalog = _catalog()
    bad = catalog.isel(polarization=slice(0, 1))

    with pytest.raises(ValueError, match="polarization"):
        validate_catalog(bad, label="test")
