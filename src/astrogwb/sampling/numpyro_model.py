from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from astrogwb.gwb import gaussian_bin_scale, spectral_density


def numpyro_model(
    *,
    frequencies,
    polarization_power,
    samples,
    observed_spectral_density,
    effective_psd,
    observation_time,
    average_mode,
    log_importance_weights_fn,
    merger_rate_fn,
    priors=None,
    constants=None,
    frequency_mask=None,
):
    priors = priors or {}
    constants = constants or {}

    sampled_params = {
        name: numpyro.sample(name, prior) for name, prior in priors.items()
    }
    params = {**constants, **sampled_params}

    log_weights = log_importance_weights_fn(params, samples)
    weights = jnp.exp(log_weights)
    total_merger_rate = merger_rate_fn(params, observation_time=observation_time)
    model_spectral_density = spectral_density(
        polarization_power,
        weights,
        total_merger_rate,
        average_mode=average_mode,
    )

    relative_ess = jnp.sum(weights) ** 2 / (weights.shape[0] * jnp.sum(weights**2))
    numpyro.deterministic("total_merger_rate", total_merger_rate)
    numpyro.deterministic("importance_relative_ess", relative_ess)

    scale = gaussian_bin_scale(effective_psd, frequencies, observation_time)
    if frequency_mask is not None:
        model_spectral_density = model_spectral_density[frequency_mask]
        observed_spectral_density = observed_spectral_density[frequency_mask]
        scale = scale[frequency_mask]

    numpyro.sample(
        "spectral_density_obs",
        dist.Normal(model_spectral_density, scale).to_event(1),
        obs=observed_spectral_density,
    )
