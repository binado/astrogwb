"""Serializable configuration for a NumPyro distribution."""

from __future__ import annotations

import numpyro.distributions as dist
import pytest

from astrogwb.distributions.config import DistributionConfig


@pytest.mark.parametrize(
    "config",
    [
        DistributionConfig(dist="Uniform", kwargs={"low": 20.0, "high": 140.0}),
        DistributionConfig(dist="Normal", kwargs={"loc": 770.0, "scale": 7.7}),
    ],
)
def test_distribution_config_round_trips_through_a_live_distribution(
    config: DistributionConfig,
) -> None:
    assert DistributionConfig.from_distribution(config.build()) == config


def test_distribution_config_builds_the_named_distribution() -> None:
    prior = DistributionConfig(dist="Uniform", kwargs={"low": 1.0, "high": 3.0}).build()

    assert isinstance(prior, dist.Uniform)
    assert float(prior.mean) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (DistributionConfig(dist="Mapping", kwargs={}), "not a numpyro distribution"),
        (DistributionConfig(dist="Uniform", kwargs={"low": 1.0}), "missing required"),
        (
            DistributionConfig(
                dist="Normal", kwargs={"loc": 0.0, "scale": 1.0, "low": 0.0}
            ),
            "Extra inputs",
        ),
    ],
)
def test_distribution_config_with_bad_name_or_kwargs_raises_value_error(
    config: DistributionConfig, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        config.build()


def test_distribution_config_rejects_a_positional_args_form() -> None:
    with pytest.raises(ValueError, match="args"):
        DistributionConfig.model_validate(
            {"dist": "Uniform", "kwargs": {"low": 0.0, "high": 1.0}, "args": [0, 1]}
        )
