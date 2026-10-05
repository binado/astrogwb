"""Tests for the population record both catalog formats carry."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from astrogwb.populations import ComponentMetadata, PopulationMetadata

REDSHIFT_KWARGS = {"minimum_redshift": 0.1, "maximum_redshift": 10.0, "n_grid": 32}


def _record(**overrides: Any) -> PopulationMetadata:
    fields: dict[str, Any] = {
        "model_name": "bns_madau_dickinson",
        "model_kwargs": {"sample_inclination": True},
        "redshift": ComponentMetadata(model="madau_dickinson", kwargs=REDSHIFT_KWARGS),
        "mass": ComponentMetadata(model="ordered_uniform"),
    }
    return PopulationMetadata(**{**fields, **overrides})


def test_json_round_trip_preserves_every_field() -> None:
    record = _record()
    restored = PopulationMetadata.model_validate_json(record.model_dump_json())
    assert restored == record
    assert type(restored.redshift.kwargs["n_grid"]) is int
    assert type(restored.redshift.kwargs["minimum_redshift"]) is float
    assert restored.model_kwargs["sample_inclination"] is True


def test_component_kwargs_reject_booleans() -> None:
    with pytest.raises(ValidationError, match="must be a number"):
        ComponentMetadata(model="madau_dickinson", kwargs={"n_grid": True})


def test_check_registered_names_the_unknown_population() -> None:
    with pytest.raises(KeyError, match="no_such_population"):
        _record(model_name="no_such_population").check_registered()


def test_check_registered_names_the_unknown_sub_models() -> None:
    with pytest.raises(KeyError, match="no_such_redshift"):
        _record(
            redshift=ComponentMetadata(model="no_such_redshift", kwargs=REDSHIFT_KWARGS)
        ).check_registered()
    with pytest.raises(KeyError, match="no_such_mass"):
        _record(mass=ComponentMetadata(model="no_such_mass")).check_registered()


def test_check_registered_rejects_a_kwarg_the_component_does_not_take() -> None:
    """A record is only valid if its kwargs actually build its population."""
    record = _record(
        redshift=ComponentMetadata(
            model="madau_dickinson",
            kwargs={**REDSHIFT_KWARGS, "uniform_mixing_fraction": 0.1},
        )
    )
    with pytest.raises(TypeError, match="uniform_mixing_fraction"):
        record.check_registered()


def test_a_proposal_population_builds_with_no_merger_rate() -> None:
    record = _record(
        redshift=ComponentMetadata(
            model="madau_dickinson_uniform_guard",
            kwargs={**REDSHIFT_KWARGS, "uniform_mixing_fraction": 0.1},
        )
    )
    assert record.build().merger_rate_fn is None


def test_with_redshift_kwargs_rewrites_only_the_redshift_record() -> None:
    record = _record()
    narrowed = record.with_redshift_kwargs(minimum_redshift=0.3)
    assert narrowed.redshift.kwargs["minimum_redshift"] == 0.3
    assert narrowed.redshift.kwargs["n_grid"] == 32
    assert narrowed.mass == record.mass
    assert narrowed.model_kwargs == record.model_kwargs


def test_a_seed_is_not_part_of_the_record() -> None:
    """A seed picks a realization; it is a simulator input, not metadata."""
    with pytest.raises(ValidationError, match="seed"):
        _record(seed=7)
