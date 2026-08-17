"""Tests for the native ``PriorDistribution`` fields on ``RunConfig``.

``RunConfig.priors`` and ``RunConfig.amplitude_prior`` hold live numpyro
``Uniform``/``Normal`` distributions via a ``BeforeValidator`` +
``PlainSerializer`` pair (see ``astrogwb_paper.config.mcmc.PriorDistribution``).
These tests pin the two properties that make that safe to do at config-parse
time:

- distributions materialize from spec mappings and serialize back to the exact
  same spec, so ``save_config`` / ``config_sha256`` stay canonical; and
- neither materialization nor serialization evaluates a JAX op -- the XLA
  backend must still be uninitialized afterwards, or
  ``configure_runtime``'s ``set_host_device_count`` silently no-ops. That
  ordering can only be observed in a fresh interpreter, so the backend check
  runs in a subprocess.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import (
    build_run_config,
    materialize_prior,
    prior_to_spec,
)
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()
EXAMPLE_CONFIG = PAPER_ROOT / "configs/mcmc.example.toml"


def test_priors_materialize_to_live_distributions() -> None:
    import numpyro.distributions as dist

    config = build_run_config(load_mapping(EXAMPLE_CONFIG))

    prior = config.priors["H0"]
    assert isinstance(prior, dist.Uniform)
    assert prior.low == 20.0
    assert prior.high == 140.0


def test_prior_field_json_dump_round_trips_the_spec() -> None:
    raw = load_mapping(EXAMPLE_CONFIG)
    config = build_run_config(raw)

    dumped = config.model_dump(mode="json")["priors"]
    expected = {p: raw["priors"][p] for p in config.sampled_params}

    assert dumped == expected


def test_python_dump_serializes_priors_back_to_specs() -> None:
    """``model_dump()`` also runs the serializer (when_used="always"): revalidating
    a python dump rebuilds equal distributions from their specs."""
    config = build_run_config(load_mapping(EXAMPLE_CONFIG))

    revalidated = build_run_config(config.model_dump())

    assert prior_to_spec(revalidated.priors["H0"]) == prior_to_spec(config.priors["H0"])


def test_materialize_prior_passes_live_distributions_through() -> None:
    """An already-built distribution validates to itself (idempotent)."""
    config = build_run_config(load_mapping(EXAMPLE_CONFIG))

    prior = config.priors["H0"]

    assert materialize_prior(prior) is prior


@pytest.mark.integration
def test_config_build_and_dump_leave_the_xla_backend_uninitialized() -> None:
    """Config parse -> dump must not consume JAX's one-shot backend config.

    ``configure_runtime`` sets ``JAX_PLATFORMS`` / ``XLA_FLAGS`` and calls
    ``numpyro.set_host_device_count`` *after* config validation; each of those
    is silently ignored once the XLA backend has initialized. This spawns a
    fresh interpreter that validates and dumps a config, and only then touches
    JAX: if anything in the config path initialized the backend, the late
    ``set_host_device_count`` is a no-op and ``jax.device_count()`` stays 1.
    """
    script = r"""
import json
import sys
from pathlib import Path

from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.config.mcmc import (
    UniformPrior,
    build_run_config,
    prior_to_spec,
)

config = build_run_config(load_mapping(Path(sys.argv[1])))

# Dists are materialized and round-trip to specs -- all pre-JAX.
assert isinstance(prior_to_spec(config.priors["H0"]), UniformPrior)
specs = {
    name: prior_to_spec(prior).model_dump(mode="json")
    for name, prior in config.priors.items()
}
json.dumps(config.model_dump(mode="json"))

# Only now touch JAX. Surviving set_host_device_count proves the backend was
# still uninitialized after validation + serialization.
import numpyro

numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"backend initialized early: {jax.devices()}"
print("backend-safe:", specs["H0"])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(EXAMPLE_CONFIG)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "backend-safe: {'type': 'uniform', 'low': 20.0, 'high': 140.0}" in (
        result.stdout
    )
