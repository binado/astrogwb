from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from astrogwb.waveform.catalog import (
    DOMAIN_FREQUENCY,
    FORMAT_NAME,
    RESERVED_ATTRS,
    load_catalog,
    make_catalog,
    open_catalog,
    save_catalog,
    validate_catalog,
)


def _catalog(nsamples: int = 6, nfreq: int = 4, **kwargs):
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
        df=10.0 if nfreq == 1 else float(frequencies[1] - frequencies[0]),
        **kwargs,
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
        "df": 10.0,
    }


def test_extra_attrs_survive_a_round_trip(tmp_path: Path) -> None:
    """Provenance a producer stamps on must come back byte-for-byte."""
    extra = {
        "population_name": "madau-dickinson",
        "population_seed": 41,
        "population_samples": 32768,
        "redshift_proposal": '{"kind": "uniform_redshift", "z_min": 0.0}',
    }
    catalog = _catalog(extra_attrs=extra)
    path = tmp_path / "catalog.h5"

    save_catalog(path, catalog)
    loaded = load_catalog(path)

    for name, value in extra.items():
        assert loaded.attrs[name] == value
    # The built-ins are untouched by the merge.
    assert loaded.attrs["format_name"] == FORMAT_NAME
    assert loaded.attrs["approximant"] == "Toy"


@pytest.mark.parametrize("reserved", sorted(RESERVED_ATTRS))
def test_extra_attrs_rejects_reserved_names(reserved: str) -> None:
    with pytest.raises(ValueError, match="reserved catalog attribute"):
        _catalog(extra_attrs={reserved: "hijacked"})


@pytest.mark.parametrize("value", [{"nested": 1}, [1, 2, 3], np.arange(3), None, True])
def test_extra_attrs_rejects_non_scalars(value: object) -> None:
    """netCDF attributes are flat scalars; anything else must fail at build time."""
    with pytest.raises(TypeError, match="must be a str, int, or float scalar"):
        _catalog(extra_attrs={"provenance": value})


def test_extra_attrs_none_and_empty_leave_attrs_untouched() -> None:
    baseline = dict(_catalog().attrs)
    assert dict(_catalog(extra_attrs=None).attrs) == baseline
    assert dict(_catalog(extra_attrs={}).attrs) == baseline


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


def test_validate_catalog_rejects_missing_df() -> None:
    catalog = _catalog()
    bad = catalog.copy()
    del bad.attrs["df"]

    with pytest.raises(ValueError, match="missing required 'df'.*regenerate"):
        validate_catalog(bad, label="test")


@pytest.mark.parametrize("df", [0.0, -1.0, np.inf, np.nan, "invalid"])
def test_validate_catalog_rejects_invalid_df(df: object) -> None:
    bad = _catalog().assign_attrs(df=df)

    with pytest.raises(ValueError, match="df must be a finite positive scalar"):
        validate_catalog(bad, label="test")


def test_validate_catalog_rejects_boolean_df() -> None:
    with pytest.raises(TypeError, match="df must be a finite positive scalar"):
        validate_catalog(_catalog().assign_attrs(df=True), label="test")


def test_validate_catalog_rejects_empty_or_nonfinite_frequencies() -> None:
    catalog = _catalog()
    empty = catalog.isel(frequency=slice(0, 0))
    nonfinite = catalog.assign_coords(frequency=[10.0, 20.0, np.inf, 40.0])

    with pytest.raises(ValueError, match="at least one bin"):
        validate_catalog(empty, label="test")
    with pytest.raises(ValueError, match="frequencies must be finite"):
        validate_catalog(nonfinite, label="test")


def test_validate_catalog_rejects_nonuniform_frequencies() -> None:
    catalog = _catalog()
    bad = catalog.assign_coords(frequency=[10.0, 20.0, 31.0, 40.0])

    with pytest.raises(ValueError, match="uniformly spaced by df"):
        validate_catalog(bad, label="test")


def test_validate_catalog_accepts_float64_fft_roundoff() -> None:
    catalog = _catalog()
    frequencies = catalog.frequency.values.copy()
    frequencies[2] += 32.0 * np.finfo(np.float64).eps * frequencies[-1]

    validate_catalog(catalog.assign_coords(frequency=frequencies), label="test")


def test_validate_catalog_rejects_df_inconsistent_with_coordinates() -> None:
    bad = _catalog().assign_attrs(df=5.0)

    with pytest.raises(ValueError, match="uniformly spaced by df=5.0"):
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
            df=10.0,
        )
