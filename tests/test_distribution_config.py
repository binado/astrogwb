"""Serializable configuration for a NumPyro distribution."""

from __future__ import annotations

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


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (DistributionConfig(dist="Mapping", kwargs={}), "not a numpyro distribution"),
        (DistributionConfig(dist="Uniform", kwargs={"low": 1.0}), "missing required"),
    ],
)
def test_distribution_config_with_bad_name_or_kwargs_raises_value_error(
    config: DistributionConfig, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        config.build()
