from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from pydantic import ValidationError

from astrogwb.sampling import numpyro_model
from astrogwb.config.mcmc import build_run_config, load_config
from astrogwb.sampling.priors import build_prior

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_numpyro_model_smoke_trace() -> None:
    frequencies = jnp.array([10.0, 20.0, 30.0])
    polarization_power = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    samples = {"mass_1": jnp.array([20.0, 30.0])}

    def merger_rate_and_log_weights_fn(params, samples):
        return jnp.array(1.0), jnp.zeros(2)

    trace = handlers.trace(
        handlers.seed(
            numpyro_model,
            rng_seed=0,
        )
    ).get_trace(
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=jnp.array([16.0, 32.0, 48.0]),
        effective_psd=jnp.ones(3),
        observation_time=2.0,
        average_mode="catalog_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        frequency_mask=jnp.array([True, False, True]),
    )

    assert trace["spectral_density_obs"]["fn"].event_shape == (2,)


def test_numpyro_model_uses_combined_merger_rate_and_log_weights_callback() -> None:
    frequencies = jnp.array([10.0, 20.0])
    polarization_power = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    samples = {"sentinel": jnp.array([4.0, 6.0])}

    def merger_rate_and_log_weights_fn(params, samples):
        shared = params["scale"] + samples["sentinel"][0]
        total_merger_rate = shared
        log_weights = jnp.log(jnp.array([shared, shared + 2.0]))
        return total_merger_rate, log_weights

    trace = handlers.trace(
        handlers.seed(
            numpyro_model,
            rng_seed=0,
        )
    ).get_trace(
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=jnp.array([1.0, 2.0]),
        effective_psd=jnp.ones(2),
        observation_time=3.0,
        average_mode="catalog_inclination",
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        constants={"scale": jnp.array(2.0)},
    )

    np.testing.assert_allclose(np.asarray(trace["total_merger_rate"]["value"]), 6.0)
    np.testing.assert_allclose(
        np.asarray(trace["importance_relative_ess"]["value"]),
        49.0 / 50.0,
    )


@pytest.mark.parametrize(
    ("spec", "expected_attrs"),
    [
        ({"type": "uniform", "low": 0.0, "high": 2.0}, {"low": 0.0, "high": 2.0}),
        ({"type": "normal", "loc": 1.0, "scale": 0.5}, {"loc": 1.0, "scale": 0.5}),
    ],
)
def test_build_prior_happy_path(
    spec: dict[str, object], expected_attrs: dict[str, float]
) -> None:
    d = build_prior(spec)

    for attr, expected in expected_attrs.items():
        np.testing.assert_allclose(getattr(d, attr), expected)


def test_build_prior_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unsupported prior type"):
        build_prior({"type": "mystery", "low": 0.0, "high": 1.0})


@pytest.mark.parametrize(
    "relative_path",
    ["configs/mcmc.example.toml", "configs/mcmc.cosmology.toml"],
)
def test_shipped_mcmc_configs_fix_local_merger_rate(relative_path: str) -> None:
    config = build_run_config(load_config(REPO_ROOT / relative_path))

    assert "local_merger_rate" not in config.sampled_params
    assert config.fiducials["local_merger_rate"] == 161.0
    assert "local_merger_rate" not in config.priors
    assert config.constants["local_merger_rate"] == 161.0


def test_legacy_top_level_local_merger_rate_is_rejected() -> None:
    raw = load_config(REPO_ROOT / "configs/mcmc.example.toml")
    raw["local_merger_rate"] = raw["fiducials"]["local_merger_rate"]

    with pytest.raises(ValidationError, match="local_merger_rate"):
        build_run_config(raw)
