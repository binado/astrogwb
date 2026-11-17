"""Tests for the population record both catalog formats carry."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from astrogwb.populations import AnyPopulation, BNSCobaPopulation

SETTINGS: dict[str, Any] = {
    "model_name": "bns_coba",
    "mass_model": "uniform",
    "minimum_redshift": 0.1,
    "maximum_redshift": 10.0,
    "n_grid": 32,
}
ANY_POPULATION = TypeAdapter(AnyPopulation)


def _record(**overrides: Any) -> BNSCobaPopulation:
    return BNSCobaPopulation.model_validate({**SETTINGS, **overrides})


def test_json_round_trip_restores_the_subclass_and_its_types() -> None:
    record = _record()
    restored = ANY_POPULATION.validate_json(ANY_POPULATION.dump_json(record))
    assert restored == record
    assert type(restored) is BNSCobaPopulation
    assert type(restored.n_grid) is int
    assert type(restored.minimum_redshift) is float


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({**SETTINGS, "model_name": "no_such_population"}, "no_such_population"),
        ({**SETTINGS, "no_such_setting": 0.1}, "no_such_setting"),
        ({**SETTINGS, "mass_model": "flat"}, "mass_model"),
        ({**SETTINGS, "maximum_redshift": 0.05}, "must exceed"),
    ],
)
def test_an_invalid_record_is_refused_when_validated(
    settings: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        ANY_POPULATION.validate_python(settings)


def test_equal_records_build_the_same_population() -> None:
    assert _record().build() is _record().build()
    assert _record().build() is not _record(n_grid=64).build()


def test_replace_revalidates() -> None:
    assert _record().replace(minimum_redshift=0.2).minimum_redshift == 0.2
    with pytest.raises(ValidationError, match="must exceed"):
        _record().replace(minimum_redshift=20.0)
