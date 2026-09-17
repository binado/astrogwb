"""Tests for the population record both catalog formats carry."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from astrogwb.metadata import (
    MODEL_KWARGS_ATTR,
    MODEL_NAME_ATTR,
    SEED_ATTR,
    PopulationMetadata,
)

MODEL_KWARGS = {"minimum_redshift": 0.1, "maximum_redshift": 10.0, "n_grid": 32}


def _record(**overrides: Any) -> PopulationMetadata:
    fields: dict[str, Any] = {
        "model_name": "bns_md_cosmological",
        "model_kwargs": MODEL_KWARGS,
        "seed": 7,
    }
    return PopulationMetadata(**{**fields, **overrides})


def test_attrs_round_trip_preserves_every_field() -> None:
    record = _record()
    restored = PopulationMetadata.from_attrs(record.to_attrs(), label="test")
    assert restored == record


def test_to_attrs_sorts_mapping_keys_so_a_file_is_reproducible() -> None:
    attrs = _record(
        model_kwargs={"maximum_redshift": 10.0, "minimum_redshift": 0.1}
    ).to_attrs()
    assert attrs[MODEL_KWARGS_ATTR] == json.dumps(
        {"minimum_redshift": 0.1, "maximum_redshift": 10.0}, sort_keys=True
    )
    assert attrs[MODEL_NAME_ATTR] == "bns_md_cosmological"
    assert attrs[SEED_ATTR] == 7


def test_kwargs_are_normalized() -> None:
    record = _record(model_kwargs={"minimum_redshift": 0.1})
    assert isinstance(record.model_kwargs, dict)


def test_build_binds_construction_kwargs_to_both_callables() -> None:
    source, rate = _record().build()
    assert source.keywords == MODEL_KWARGS  # ty: ignore[unresolved-attribute]
    # One kwargs mapping reaches both: nothing is filtered on the way to the
    # rate, so a key neither accepts fails rather than being dropped.
    assert rate is not None
    assert rate.keywords == MODEL_KWARGS  # ty: ignore[unresolved-attribute]


def test_check_registered_names_the_unknown_population() -> None:
    with pytest.raises(KeyError, match="no_such_population"):
        _record(model_name="no_such_population").check_registered()


def test_check_registered_rejects_a_kwarg_the_population_does_not_take() -> None:
    """A record is only valid if its kwargs actually build its population.

    The flat kwargs mapping used to be filtered down to the shared window keys
    before reaching the rate function, so a stale key travelled unnoticed.
    """
    record = _record(model_kwargs={**MODEL_KWARGS, "uniform_mixing_fraction": 0.1})
    with pytest.raises(TypeError, match="uniform_mixing_fraction"):
        record.check_registered()


def test_a_proposal_population_builds_with_no_merger_rate() -> None:
    record = _record(
        model_name="bns_md_uniform_mixture",
        model_kwargs={**MODEL_KWARGS, "uniform_mixing_fraction": 0.1},
    )
    assert record.build().merger_rate_fn is None


@pytest.mark.parametrize("seed", ["7", 7.0, True, None])
def test_seed_must_be_a_non_boolean_int(seed: object) -> None:
    """``True`` is the case strict validation exists for.

    Pydantic's default lax mode widens a bool to an int, so a seed of
    ``True`` would validate as ``1`` and a draw would record a seed it was
    never made at.
    """
    with pytest.raises(ValidationError, match="seed"):
        _record(seed=seed)


def test_from_attrs_rejects_a_malformed_json_attribute() -> None:
    attrs = {**_record().to_attrs(), MODEL_KWARGS_ATTR: "{not json"}
    with pytest.raises(ValueError, match=MODEL_KWARGS_ATTR):
        PopulationMetadata.from_attrs(attrs, label="broken.h5")
