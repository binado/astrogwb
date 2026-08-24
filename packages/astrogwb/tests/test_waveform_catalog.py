from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from astrogwb.waveform.catalog import (
    DOMAIN_FREQUENCY,
    FORMAT_NAME,
    load_catalog,
    make_catalog,
    open_catalog,
    save_catalog,
    validate_catalog,
)


def _catalog(nsamples: int = 6, nfreq: int = 4):
    rng = np.random.default_rng(0)
    frequencies = np.linspace(10.0, 40.0, nfreq)
    power = rng.uniform(0.0, 1.0, size=(nfreq, nsamples))
    source_parameters = {
        "redshift": rng.uniform(0.0, 2.0, nsamples),
        "luminosity_distance": rng.uniform(100.0, 1000.0, nsamples),
    }
    return make_catalog(
        frequencies=frequencies,
        polarization_power=power,
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
        loaded.polarization_power.values, catalog.polarization_power.values
    )
    assert loaded.polarization_power.dtype == np.float64
    np.testing.assert_allclose(
        loaded.source_parameters.values, catalog.source_parameters.values
    )
    assert dict(loaded.attrs) == {
        "format_name": FORMAT_NAME,
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
        assert f["polarization_power"].compression == "gzip"
    loaded = load_catalog(path)
    np.testing.assert_allclose(
        loaded.polarization_power.values, catalog.polarization_power.values
    )


def test_save_catalog_reorders_all_sample_variables_in_blocks(tmp_path: Path) -> None:
    catalog = _catalog(nsamples=4).assign_coords(detector=["H1", "L1"])
    snr = np.arange(8.0).reshape(4, 2)
    catalog = catalog.assign(snr=(("sample", "detector"), snr))
    order = np.array([2, 0, 3, 1])
    path = tmp_path / "catalog.h5"

    save_catalog(path, catalog, sample_order=order)
    loaded = load_catalog(path)

    np.testing.assert_array_equal(loaded.detector.values, ["H1", "L1"])
    np.testing.assert_allclose(
        loaded.polarization_power.values,
        catalog.polarization_power.values[:, order],
    )
    np.testing.assert_allclose(
        loaded.source_parameters.values,
        catalog.source_parameters.values[order],
    )
    np.testing.assert_allclose(loaded.snr.values, snr[order])


def test_reordered_save_preserves_existing_compression(tmp_path: Path) -> None:
    source = tmp_path / "source.h5"
    destination = tmp_path / "destination.h5"
    save_catalog(source, _catalog(nsamples=3), compression="gzip")

    with open_catalog(source) as catalog:
        save_catalog(destination, catalog, sample_order=[2, 1, 0])

    with h5py.File(destination) as output:
        assert output["polarization_power"].compression == "gzip"


def test_save_catalog_rejects_incomplete_sample_permutation(tmp_path: Path) -> None:
    catalog = _catalog(nsamples=3)

    with pytest.raises(ValueError, match="complete permutation"):
        save_catalog(tmp_path / "catalog.h5", catalog, sample_order=[0, 0, 2])


def test_validate_catalog_rejects_invalid_snr() -> None:
    catalog = _catalog(nsamples=2).assign_coords(detector=["H1", "H1"])
    bad = catalog.assign(
        snr=(("sample", "detector"), np.array([[1.0, 2.0], [3.0, -1.0]]))
    )

    with pytest.raises(ValueError, match="detector names"):
        validate_catalog(bad, label="test")


def test_open_catalog_leaves_polarization_power_lazy(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.h5"
    save_catalog(path, catalog)

    opened = open_catalog(path)

    assert opened.polarization_power._in_memory is False


def test_load_catalog_rejects_wrong_format_name(tmp_path: Path) -> None:
    catalog = _catalog()
    path = tmp_path / "catalog.h5"
    save_catalog(path, catalog)
    with h5py.File(path, "r+") as f:
        f.attrs["format_name"] = "something_else"

    with pytest.raises(ValueError, match="format_name"):
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


def test_validate_catalog_rejects_missing_polarization_power() -> None:
    catalog = _catalog()
    bad = catalog.drop_vars("polarization_power")

    with pytest.raises(ValueError, match="polarization_power"):
        validate_catalog(bad, label="test")


def test_validate_catalog_rejects_wrong_dims() -> None:
    catalog = _catalog()
    bad = catalog.rename_dims({"sample": "event"})

    with pytest.raises(ValueError, match="frequency, sample"):
        validate_catalog(bad, label="test")


def test_make_catalog_rejects_complex_polarization_power() -> None:
    frequencies = np.linspace(10.0, 40.0, 4)
    power = np.zeros((4, 3), dtype=np.complex128)
    source_parameters = {"redshift": np.array([0.1, 0.5, 1.0])}

    with pytest.raises(ValueError, match="real-valued"):
        make_catalog(
            frequencies=frequencies,
            polarization_power=power,  # ty: ignore[invalid-argument-type]
            source_parameters=source_parameters,
            approximant="Toy",
            minimum_frequency=10.0,
            maximum_frequency=40.0,
            reference_frequency=20.0,
            sampling_frequency=128.0,
        )
