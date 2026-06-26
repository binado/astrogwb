from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from astrogwb.gwb import AverageMode, gaussian_bin_scale, spectral_density


class MergerRateAndLogWeightsFn(Protocol):
    def __call__(
        self,
        params: Mapping[str, Any],
        samples: Mapping[str, jax.Array],
    ) -> tuple[float | jax.Array, jax.Array]: ...


def numpyro_model(
    *,
    frequencies: jax.Array,
    polarization_power: jax.Array,
    samples: Mapping[str, jax.Array],
    observed_spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time: float,
    average_mode: AverageMode,
    merger_rate_and_log_weights_fn: MergerRateAndLogWeightsFn,
    priors: Mapping[str, dist.Distribution] | None = None,
    constants: Mapping[str, Any] | None = None,
    frequency_mask: jax.Array | None = None,
) -> None:
    """NumPyro model for importance-weighted SGWB inference.

    Parameters
    ----------
    observation_time:
        Observation time in years, used only in the likelihood noise scale.
    """
    priors = priors or {}
    constants = constants or {}

    sampled_params = {
        name: numpyro.sample(name, prior) for name, prior in priors.items()
    }
    params = {**constants, **sampled_params}

    total_merger_rate, log_weights = merger_rate_and_log_weights_fn(
        params,
        samples,
    )
    weights = jnp.exp(log_weights)
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
