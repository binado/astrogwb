from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro import handlers

from astrogwb.sampling import numpyro_model


def test_numpyro_model_smoke_trace() -> None:
    frequencies = jnp.array([10.0, 20.0, 30.0])
    polarization_power = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    samples = {"mass_1": jnp.array([20.0, 30.0])}

    def log_importance_weights_fn(params, samples):
        return jnp.log(jnp.array([1.0, params["rate_scale"]]))

    def merger_rate_fn(params, *, observation_time):
        return params["rate_scale"] * observation_time

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
        log_importance_weights_fn=log_importance_weights_fn,
        merger_rate_fn=merger_rate_fn,
        priors={"rate_scale": dist.Delta(jnp.array(2.0))},
        constants={"rate_scale": jnp.array(100.0)},
        frequency_mask=jnp.array([True, False, True]),
    )

    assert trace["spectral_density_obs"]["fn"].event_shape == (2,)
    np.testing.assert_allclose(np.asarray(trace["total_merger_rate"]["value"]), 4.0)
    np.testing.assert_allclose(
        np.asarray(trace["importance_relative_ess"]["value"]),
        9.0 / 10.0,
    )
