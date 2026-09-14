"""Tests for the population record both catalog formats carry."""

from __future__ import annotations

import json
from typing import Any

import pytest

from astrogwb.populations import PopulationRecord
from astrogwb.populations.record import (
    DENSITY_SITES_ATTR,
    MODEL_KWARGS_ATTR,
    RATE_MODEL_NAME_ATTR,
    SEED_ATTR,
    SOURCE_MODEL_NAME_ATTR,
)

MODEL_KWARGS = {"z_min": 0.1, "z_max": 10.0, "n_grid": 32}
DENSITY_SITES = ("redshift", "source_frame_mass_1", "source_frame_mass_2")


def _record(**overrides: Any) -> PopulationRecord:
    fields: dict[str, Any] = {
        "source_model_name": "bns_md_cosmological",
        "rate_model_name": "madau_dickinson",
        "model_kwargs": MODEL_KWARGS,
        "density_sites": DENSITY_SITES,
        "seed": 7,
    }
    return PopulationRecord(**{**fields, **overrides})


def test_attrs_round_trip_preserves_every_field() -> None:
    record = _record()
    restored = PopulationRecord.from_attrs(record.to_attrs(), label="test")
    assert restored == record


def test_to_attrs_sorts_mapping_keys_so_a_file_is_reproducible() -> None:
    attrs = _record(model_kwargs={"z_max": 10.0, "z_min": 0.1}).to_attrs()
    assert attrs[MODEL_KWARGS_ATTR] == json.dumps(
        {"z_min": 0.1, "z_max": 10.0}, sort_keys=True
    )
    assert attrs[DENSITY_SITES_ATTR] == json.dumps(list(DENSITY_SITES))
    assert attrs[SOURCE_MODEL_NAME_ATTR] == "bns_md_cosmological"
    assert attrs[RATE_MODEL_NAME_ATTR] == "madau_dickinson"
    assert attrs[SEED_ATTR] == 7


def test_density_sites_and_kwargs_are_normalized() -> None:
    record = _record(density_sites=["redshift"], model_kwargs={"z_min": 0.1})
    assert record.density_sites == ("redshift",)
    assert isinstance(record.model_kwargs, dict)


def test_getters_bind_construction_settings_only() -> None:
    record = _record()
    source = record.get_source_model()
    rate = record.get_merger_rate_fn()
    assert source.keywords == MODEL_KWARGS  # ty: ignore[unresolved-attribute]
    # Only the shared window/grid keys reach the rate function.
    assert rate.keywords == MODEL_KWARGS  # ty: ignore[unresolved-attribute]


def test_check_registered_names_the_unknown_source_model() -> None:
    with pytest.raises(KeyError, match="no_such_population"):
        _record(source_model_name="no_such_population").check_registered()


def test_check_registered_names_the_unknown_rate_model() -> None:
    with pytest.raises(KeyError, match="no_such_rate"):
        _record(rate_model_name="no_such_rate").check_registered()


@pytest.mark.parametrize("seed", ["7", 7.0, True, None])
def test_seed_must_be_a_non_boolean_int(seed: object) -> None:
    with pytest.raises(TypeError, match="seed"):
        _record(seed=seed)


def test_from_attrs_rejects_a_malformed_json_attribute() -> None:
    attrs = {**_record().to_attrs(), MODEL_KWARGS_ATTR: "{not json"}
    with pytest.raises(ValueError, match=MODEL_KWARGS_ATTR):
        PopulationRecord.from_attrs(attrs, label="broken.h5")
