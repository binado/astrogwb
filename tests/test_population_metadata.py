"""Tests for the population record both catalog formats carry."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from numpyro import handlers
from pydantic import TypeAdapter, ValidationError

from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import AnyPopulation, BNSCobaPopulation
from astrogwb.simulators import RealizationMetadata

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
        ({**SETTINGS, "time_delay": True}, "requires minimum_delay"),
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


def _draw_metadata() -> RealizationMetadata:
    return RealizationMetadata(
        population=_record(),
        hyperparameters={
            "H0": 67.66,
            "gamma": DistributionConfig(
                dist="Uniform", kwargs={"low": 1.0, "high": 2.0}
            ),
        },
        observation_time=1.0,
    )


def test_prior_model_has_a_scalar_site_per_hyperparameter() -> None:
    metadata = _draw_metadata()
    model = metadata.prior_model()
    trace = handlers.trace(handlers.seed(model, jax.random.key(0))).get_trace()
    sampled = {n for n, s in trace.items() if s["type"] == "sample"}
    deterministic = {n for n, s in trace.items() if s["type"] == "deterministic"}
    assert sampled == set(metadata.sampled)
    assert deterministic == set(metadata.fixed)
    assert trace["H0"]["value"].dtype == jnp.float64
    assert float(trace["H0"]["value"]) == 67.66

    theta = handlers.seed(model, jax.random.key(3))()
    assert set(theta) == {"H0", "gamma"}
    assert all(v.shape == () for v in theta.values())
    assert 1.0 <= float(theta["gamma"]) <= 2.0
