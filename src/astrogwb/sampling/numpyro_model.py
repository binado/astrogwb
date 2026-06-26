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

    Samples hyperparameters from ``priors``, evaluates a fixed proposal catalog
    through ``merger_rate_and_log_weights_fn``, and compares the predicted
    stochastic gravitational-wave background (SGWB) spectral density to
    ``observed_spectral_density`` under a per-frequency Gaussian likelihood.

    The predicted spectrum is built from precomputed per-source polarization
    powers and importance weights ``exp(log_weights)``; waveform generation is
    not part of this model. ``constants`` are merged with sampled parameters
    before the callback is invoked.

    Registered sites:

    - one ``numpyro.sample`` per entry in ``priors``;
    - ``total_merger_rate`` and ``importance_relative_ess`` as deterministics;
    - ``spectral_density_obs`` as the observed Gaussian likelihood.

    Parameters
    ----------
    frequencies:
        Frequency grid in Hz, shape ``(F,)``.
    polarization_power:
        Per-source polarization power at each frequency, shape ``(F, N)`` where
        ``N`` is the catalog size.
    samples:
        Catalog arrays passed to ``merger_rate_and_log_weights_fn``. Each value
        should have leading dimension ``N``.
    observed_spectral_density:
        Observed SGWB spectral density at ``frequencies``, shape ``(F,)`` (or
        the masked subset when ``frequency_mask`` is set).
    effective_psd:
        Network effective power spectral density at ``frequencies``, shape
        ``(F,)``.
    observation_time:
        Observation time in years, used only in the likelihood noise scale via
        :func:`astrogwb.gwb.gaussian_bin_scale`.
    average_mode:
        How inclination is averaged when contracting polarization power:
        ``"analytic_inclination"`` applies the usual 0.4 factor;
        ``"catalog_inclination"`` uses the catalog weights directly.
    merger_rate_and_log_weights_fn:
        Callable ``(params, samples) -> (total_merger_rate, log_weights)``.
        ``params`` merges ``constants`` with sampled values; ``log_weights``
        has shape ``(N,)``. ``total_merger_rate`` is in mergers per second.
    priors:
        Mapping from parameter name to NumPyro prior distribution. Keys become
        sampled sites; defaults to an empty mapping (likelihood-only model).
    constants:
        Fixed parameter values merged into ``params`` for the callback. Defaults
        to an empty mapping.
    frequency_mask:
        Optional boolean mask of shape ``(F,)``. When provided, only masked
        frequency bins enter the likelihood.
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
