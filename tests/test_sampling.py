from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from astrogwb.sampling import numpyro_model
from astrogwb.sampling.priors import build_prior, build_priors


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


def test_build_prior_uniform() -> None:
    d = build_prior({"type": "uniform", "low": 0.0, "high": 2.0})
    assert d.low == 0.0  # ty: ignore[unresolved-attribute]
    assert d.high == 2.0  # ty: ignore[unresolved-attribute]


def test_build_prior_normal() -> None:
    d = build_prior({"type": "normal", "loc": 1.0, "scale": 0.5})
    np.testing.assert_allclose(d.loc, 1.0)  # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(d.scale, 0.5)  # ty: ignore[unresolved-attribute]


def test_build_prior_loguniform_bounds() -> None:
    import jax

    d = build_prior({"type": "loguniform", "low": 1.0, "high": 100.0})
    # LogUniform samples are positive and within [low, high].
    key = jax.random.PRNGKey(0)
    samples = np.asarray(d.sample(key=key, sample_shape=(8,)))
    assert np.all(samples >= 1.0)
    assert np.all(samples <= 100.0)


def test_build_prior_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unsupported prior type"):
        build_prior({"type": "mystery", "low": 0.0, "high": 1.0})


def test_build_priors_mapping() -> None:
    priors = build_priors(
        {
            "H0": {"type": "uniform", "low": 20.0, "high": 140.0},
            "Omega_m": {"type": "normal", "loc": 0.3, "scale": 0.1},
        }
    )
    assert set(priors) == {"H0", "Omega_m"}
    assert priors["H0"].low == 20.0  # ty: ignore[unresolved-attribute]
    assert priors["H0"].high == 140.0  # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(priors["Omega_m"].loc, 0.3)  # ty: ignore[unresolved-attribute]
