"""Round-trip and format tests for the catalog's own HDF5 serialization.

A catalog file records the density that drew it: the registered population
model, that model's construction settings, the hyperparameters, and the
included density factors. Everything here is about that record surviving the
round trip intact -- and about a file that no longer describes its own arrays
being rejected rather than loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from catalog_fixtures import (
    PAPER_MODEL,
    PAPER_MODEL_KWARGS,
    PAPER_POPULATION_PARAMS,
    make_catalog,
    save_catalog,
)

from astrogwb.catalog import Catalog
from astrogwb.catalog._io import (
    DENSITY_SITES_ATTR,
    DOMAIN_FREQUENCY,
    FORMAT_NAME,
    MODEL_KWARGS_ATTR,
    MODEL_NAME_ATTR,
    POPULATION_PARAMS_ATTR,
    PROPOSAL_ATTR,
    RESERVED_ATTRS,
    catalog_to_dataset,
    validate_catalog_dataset,
)
from astrogwb.waveform import PolarizationPowerGenerator

REDSHIFT = np.array([0.1, 0.5, 1.0])


def _catalog(**overrides) -> Catalog:
    return make_catalog(redshift=REDSHIFT, num_frequencies=4, **overrides)


def test_catalog_to_dataset_uses_stacked_float64_format() -> None:
    dataset = catalog_to_dataset(_catalog(provenance={"producer": "test"}))

    assert dataset.polarization_power.dims == ("frequency", "sample")
    assert dataset.source_parameters.dims == ("sample", "parameter")
    assert dataset.source_parameters.dtype == np.float64
    assert "redshift" in dataset.parameter.values.tolist()
    assert dataset.attrs["format_name"] == FORMAT_NAME
    assert dataset.attrs["domain"] == DOMAIN_FREQUENCY
    assert dataset.attrs["producer"] == "test"


def test_the_population_record_travels_as_data_not_as_a_callable() -> None:
    """A file stores a name and settings; the model is rebuilt from the registry."""
    dataset = catalog_to_dataset(_catalog())

    assert dataset.attrs[MODEL_NAME_ATTR] == PAPER_MODEL
    assert json.loads(dataset.attrs[MODEL_KWARGS_ATTR]) == PAPER_MODEL_KWARGS
    assert json.loads(dataset.attrs[POPULATION_PARAMS_ATTR]) == (
        PAPER_POPULATION_PARAMS
    )
    assert json.loads(dataset.attrs[DENSITY_SITES_ATTR]) == sorted(("redshift",))
    for value in dataset.attrs.values():
        assert isinstance(value, str | int | float)


def test_round_trip_preserves_arrays_and_the_population_record(
    tmp_path: Path,
) -> None:
    original = _catalog(provenance={"producer": "test", "version": 2})
    path = tmp_path / "catalog.h5"
    original.save(path)

    restored = Catalog.load(path)

    assert type(restored.waveform_metadata) is PolarizationPowerGenerator
    np.testing.assert_array_equal(
        restored.waveform_metadata.frequencies,
        original.waveform_metadata.frequencies,
    )
    np.testing.assert_array_equal(
        restored.polarization_power, original.polarization_power
    )
    assert set(restored.source_parameters) == set(original.source_parameters)
    for name, values in original.source_parameters.items():
        np.testing.assert_allclose(restored.source_parameters[name], values)
        assert restored.source_parameters[name].dtype == np.float64

    assert restored.population_model_name == original.population_model_name
    assert restored.population_model_kwargs == original.population_model_kwargs
    assert restored.population_params == original.population_params
    assert restored.density_sites == original.density_sites
    assert restored.population_metadata.seed == 41
    assert restored.population_metadata.source_type == "bns"
    assert restored.population_metadata.provenance == {
        "producer": "test",
        "version": 2,
    }


def test_a_loaded_catalog_reconstructs_its_model_and_evaluates_at_new_params(
    tmp_path: Path,
) -> None:
    """The point of the record: the density is recoverable, not just described."""
    from astrogwb.populations import (
        BNSMadauDickinson,
    )

    path = tmp_path / "catalog.h5"
    _catalog().save(path)
    restored = Catalog.load(path)

    model = restored.get_population_model()
    assert isinstance(model, BNSMadauDickinson)
    assert {
        name: getattr(model, name) for name in ("z_min", "z_max", "n_grid")
    } == PAPER_MODEL_KWARGS

    for params in (
        restored.population_params,
        {**restored.population_params, "H0": 74.0},
    ):
        values = restored.source_parameters
        site_log_probs, _ = model.evaluate(params, values)
        assert site_log_probs.shape == REDSHIFT.shape


def test_save_with_compression_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "catalog.h5"
    save_catalog(path, _catalog(), compression="gzip")

    with h5py.File(path) as handle:
        assert handle["polarization_power"].compression == "gzip"
    np.testing.assert_array_equal(
        Catalog.load(path).polarization_power, _catalog().polarization_power
    )


@pytest.mark.parametrize("reserved", sorted(RESERVED_ATTRS))
def test_provenance_rejects_reserved_names(reserved: str) -> None:
    with pytest.raises(ValueError, match="reserved catalog attribute"):
        catalog_to_dataset(_catalog(provenance={reserved: "hijacked"}))


def test_optional_source_type_is_omitted_and_decodes_as_none(tmp_path: Path) -> None:
    path = tmp_path / "catalog.h5"
    _catalog(source_type=None).save(path)

    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        assert "population_source_type" not in dataset.attrs
    assert Catalog.load(path).population_metadata.source_type is None


@pytest.mark.parametrize(
    "legacy", ["waveform_catalog", "astrogwb_catalog", "astrogwb_catalog_v2"]
)
def test_files_without_a_population_record_require_regeneration(
    tmp_path: Path, legacy: str
) -> None:
    """No legacy reader: a file that cannot say what drew it is not loadable."""
    path = tmp_path / "old.h5"
    dataset = catalog_to_dataset(_catalog())
    dataset.attrs["format_name"] = legacy
    dataset.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="regenerate"):
        Catalog.load(path)


@pytest.mark.parametrize(
    "attribute",
    [MODEL_NAME_ATTR, MODEL_KWARGS_ATTR, POPULATION_PARAMS_ATTR, DENSITY_SITES_ATTR],
)
def test_a_missing_population_attribute_requires_regeneration(
    tmp_path: Path, attribute: str
) -> None:
    path = tmp_path / "incomplete.h5"
    dataset = catalog_to_dataset(_catalog())
    del dataset.attrs[attribute]
    dataset.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="regenerate"):
        Catalog.load(path)


def test_a_model_whose_density_moved_is_caught_on_load(tmp_path: Path) -> None:
    """A registry key pins a name, not the mathematics behind it.

    The recorded fingerprint is the generating redshift density at fixed probe
    points, recomputed on every load. Shifting the recorded parameters stands in
    for a registered model whose density changed underneath an existing file --
    the case the name alone can never catch.
    """
    path = tmp_path / "drifted.h5"
    dataset = catalog_to_dataset(_catalog())
    moved = {**PAPER_POPULATION_PARAMS, "gamma": 2.7}
    dataset.attrs[POPULATION_PARAMS_ATTR] = json.dumps(moved, sort_keys=True)
    dataset.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="no longer reproduces the redshift density"):
        Catalog.load(path)


def test_a_stored_column_that_drifted_from_the_population_is_caught(
    tmp_path: Path,
) -> None:
    """Derived columns are recomputed and compared, not trusted."""
    path = tmp_path / "drifted.h5"
    catalog = _catalog()
    corrupted = dict(catalog.source_parameters)
    corrupted["detector_frame_mass_1"] = corrupted["detector_frame_mass_1"] * 1.01
    dataset = catalog_to_dataset(
        Catalog(
            source_parameters=corrupted,
            polarization_power=catalog.polarization_power,
            waveform_metadata=catalog.waveform_metadata,
            population_metadata=catalog.population_metadata,
            _model_name=catalog.population_model_name,
            _model_kwargs=catalog.population_model_kwargs,
            _population_params=catalog.population_params,
            _density_sites=catalog.density_sites,
        )
    )
    dataset.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="detector_frame_mass_1"):
        Catalog.load(path)


def test_an_unregistered_model_name_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "unknown.h5"
    dataset = catalog_to_dataset(_catalog())
    dataset.attrs[MODEL_NAME_ATTR] = "no_such_population"
    dataset.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(KeyError, match="bns_md_cosmological"):
        Catalog.load(path)


def test_a_corrupt_drift_fingerprint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.h5"
    dataset = catalog_to_dataset(_catalog())
    dataset.attrs[PROPOSAL_ATTR] = "not json"
    dataset.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="not valid JSON"):
        Catalog.load(path)


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
    _catalog().save(path)

    with xr.open_dataset(path, engine="h5netcdf") as dataset:
        validate_catalog_dataset(dataset, label="test")
        assert dataset.polarization_power._in_memory is False


def test_in_memory_catalogs_do_not_need_the_io_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional-dependency boundary: xarray is imported by save/load only."""
    import sys

    monkeypatch.setitem(sys.modules, "astrogwb.catalog._io", None)
    catalog = _catalog()
    assert catalog.restrict_redshift(0.3, 20.0).population_metadata.num_samples == 2
    assert catalog.get_population_model() is not None


def test_round_trip_restores_an_ordered_nondefault_density_selection(
    tmp_path: Path,
) -> None:
    sites = ("spin_1z", "redshift")
    catalog = _catalog(density_sites=sites)
    path = tmp_path / "selected-density.h5"
    catalog.save(path)
    restored = Catalog.load(path)
    assert restored.density_sites == sites
    model = restored.get_population_model()
    assert model.density_sites == sites
    np.testing.assert_array_equal(
        model.log_prob(restored.population_params, restored.source_parameters),
        catalog.get_population_model().log_prob(
            catalog.population_params, catalog.source_parameters
        ),
    )
    assert (
        restored.restrict_redshift(0.2, 2.0).get_population_model().density_sites
        == sites
    )
