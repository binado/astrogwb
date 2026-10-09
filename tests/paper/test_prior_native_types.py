"""Tests for prior specs: materialization into numpyro and back.

The backend-safety half (materializing priors must not initialize the XLA
backend) lives in ``test_cli.py``, since it can only be observed in a fresh
interpreter.
"""

from __future__ import annotations

import numpyro.distributions as dist
import pytest
from repo import REPO_ROOT

from astrogwb.paper.config import priors
from astrogwb.paper.config.priors import materialize_prior, prior_to_spec


def test_priors_materialize_to_live_distributions() -> None:
    assert isinstance(priors(REPO_ROOT)["H0"], dist.Uniform)
    assert isinstance(priors(REPO_ROOT)["Omega_m"], dist.Normal)


@pytest.mark.parametrize(
    "spec",
    [
        {"dist": "Uniform", "kwargs": {"low": 20.0, "high": 140.0}},
        {"dist": "Normal", "kwargs": {"loc": 0.3, "scale": 0.006}},
    ],
)
def test_a_spec_round_trips_through_a_distribution(spec: dict) -> None:
    assert prior_to_spec(materialize_prior(spec)) == spec


def test_materialize_prior_passes_live_distributions_through() -> None:
    """An already-built distribution validates to itself (idempotent)."""
    prior = priors(REPO_ROOT)["H0"]

    assert materialize_prior(prior) is prior
