"""The prior record a spectral-density draw carries for a sampled hyperparameter."""

from __future__ import annotations

import numpyro.distributions as dist
import pytest

from astrogwb.metadata import PriorSpec


@pytest.mark.parametrize(
    "spec",
    [
        PriorSpec(dist="Uniform", kwargs={"low": 20.0, "high": 140.0}),
        PriorSpec(dist="Normal", kwargs={"loc": 770.0, "scale": 7.7}),
    ],
)
def test_prior_spec_round_trips_through_a_live_distribution(spec: PriorSpec) -> None:
    assert PriorSpec.from_distribution(spec.build()) == spec


def test_prior_spec_builds_the_named_distribution() -> None:
    prior = PriorSpec(dist="Uniform", kwargs={"low": 1.0, "high": 3.0}).build()

    assert isinstance(prior, dist.Uniform)
    assert float(prior.mean) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (PriorSpec(dist="Mapping", kwargs={}), "not a numpyro distribution"),
        (PriorSpec(dist="Uniform", kwargs={"low": 1.0}), "missing required"),
        (
            PriorSpec(dist="Normal", kwargs={"loc": 0.0, "scale": 1.0, "low": 0.0}),
            "Extra inputs",
        ),
    ],
)
def test_prior_spec_with_bad_name_or_kwargs_raises_value_error(
    spec: PriorSpec, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        spec.build()


def test_prior_spec_rejects_a_positional_args_form() -> None:
    with pytest.raises(ValueError, match="args"):
        PriorSpec.model_validate(
            {"dist": "Uniform", "kwargs": {"low": 0.0, "high": 1.0}, "args": [0, 1]}
        )
