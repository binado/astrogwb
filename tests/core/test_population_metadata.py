"""Tests for the population record both catalog formats carry."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from astrogwb.populations import PopulationMetadata

MODEL_KWARGS = {"minimum_redshift": 0.1, "maximum_redshift": 10.0, "n_grid": 32}


def _record(**overrides: Any) -> PopulationMetadata:
    fields: dict[str, Any] = {
        "model_name": "bns_md_cosmological",
        "model_kwargs": MODEL_KWARGS,
    }
    return PopulationMetadata(**{**fields, **overrides})


def test_json_round_trip_preserves_every_field() -> None:
    record = _record()
    restored = PopulationMetadata.model_validate_json(record.model_dump_json())
    assert restored == record
    assert type(restored.model_kwargs["n_grid"]) is int
    assert type(restored.model_kwargs["minimum_redshift"]) is float


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


def test_a_seed_is_not_part_of_the_record() -> None:
    """A seed picks a realization; it is a simulator input, not metadata."""
    with pytest.raises(ValidationError, match="seed"):
        _record(seed=7)
