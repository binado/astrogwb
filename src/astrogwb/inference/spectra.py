"""Draw many forward-model spectra, with hyperparameters fixed or sampled.

:func:`~astrogwb.inference.gwb_forward_model` is one draw at one set of
hyperparameters. :func:`draw_spectral_density` is the loop every consumer had
been writing around it: size the static event plate from the Poisson mean, run
:class:`~numpyro.infer.Predictive`, and pull the arrays out. It works on live
objects -- a source model, a rate, a generator -- and records nothing, so a
caller free to wrap its source model (in
:class:`~astrogwb.populations.IsotropicInclination`, say) can still use it.
The recorded, cached path is :class:`astrogwb.catalog.SpectrumGenerator`,
which builds those objects from a :class:`~astrogwb.metadata.SpectraMetadata`
and calls this.

A hyperparameter is either a number, shared by every draw, or a numpyro
distribution it is drawn from once per draw. The draw runs in three stages:

1. Split ``rng_key`` into a hyperparameter key and a forward-model key, and
   draw the sampled hyperparameters with the first.
2. ``max_events`` sizes a plate, so it must be one static integer: evaluate the
   merger rate at every draw's hyperparameters and size the plate from the
   *largest* Poisson mean. With every hyperparameter fixed that is the one mean.
3. Run the forward model with the second key, replaying stage 1's values as
   sample sites.

Splitting the key up front is what keeps the forward model's randomness
independent of which hyperparameters are sampled, and what lets two calls with
the same key -- one per waveform approximant, say -- replay the same events.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpy.typing import NDArray
from numpyro.infer import Predictive

from astrogwb.populations import MergerRateFn, SourceFn
from astrogwb.inference.forward_model import gwb_forward_model
from astrogwb.utils import years_to_seconds
from astrogwb.waveform import PolarizationPowerGenerator

__all__ = [
    "SpectralDensityDraws",
    "draw_spectral_density",
    "padded_event_capacity",
]

#: The forward-model sites every draw returns.
_RETURN_SITES = ("spectral_density", "n_events", "total_merger_rate")


@dataclass(frozen=True, slots=True)
class SpectralDensityDraws:
    """Forward-model draws as NumPy arrays, draw-first.

    ``spectral_density`` has shape ``(draws, F)``; ``n_events``,
    ``total_merger_rate`` and every ``hyperparameters`` column have shape
    ``(draws,)``. A fixed hyperparameter's column is that value repeated.
    """

    frequencies: NDArray[np.floating[Any]]
    spectral_density: NDArray[np.floating[Any]]
    n_events: NDArray[np.int64]
    total_merger_rate: NDArray[np.float64]
    hyperparameters: dict[str, NDArray[np.float64]]


def padded_event_capacity(mean_count: float, n_max_sigma: float) -> int:
    """Static plate size: the Poisson mean plus ``n_max_sigma`` standard deviations.

    A draw above it is capped, as :func:`gwb_forward_model` documents; at the
    default five sigma that is a one-in-a-few-million event per draw.
    """
    if mean_count < 0.0:
        raise ValueError("Poisson mean must be non-negative")
    if n_max_sigma < 0.0:
        raise ValueError("n_max_sigma must be non-negative")
    return max(int(np.ceil(mean_count + n_max_sigma * np.sqrt(mean_count))), 1)


def draw_spectral_density(
    *,
    source_model: SourceFn,
    merger_rate_fn: MergerRateFn,
    generator: PolarizationPowerGenerator,
    hyperparameters: Mapping[str, float | dist.Distribution],
    observation_time: float,
    num_draws: int,
    rng_key: jax.Array,
    batch_size: int = 128,
    n_max_sigma: float = 5.0,
) -> SpectralDensityDraws:
    """Draw ``num_draws`` spectra from the forward model.

    ``observation_time`` is in years, as :func:`gwb_forward_model` takes it.
    Enable ``jax_enable_x64`` before calling: the rate evaluation and the draws
    otherwise run in float32.
    """
    if num_draws <= 0:
        raise ValueError("num_draws must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if observation_time <= 0.0:
        raise ValueError("observation_time must be positive")

    fixed = {
        name: float(value)
        for name, value in hyperparameters.items()
        if not isinstance(value, dist.Distribution)
    }
    priors = {
        name: value
        for name, value in hyperparameters.items()
        if isinstance(value, dist.Distribution)
    }
    theta_key, forward_key = jax.random.split(rng_key)

    # Stage 1: the hyperparameters, one row per draw.
    sampled: dict[str, jax.Array] = {}
    if priors:

        def prior_model() -> None:
            for name, prior in priors.items():
                numpyro.sample(name, prior)

        sampled = Predictive(prior_model, num_samples=num_draws)(theta_key)
    columns = {
        **{name: jnp.full(num_draws, value) for name, value in fixed.items()},
        **sampled,
    }

    # Stage 2: one static plate, deep enough for the busiest draw.
    rates = jax.vmap(lambda theta: jnp.reshape(merger_rate_fn(theta), ()))(columns)
    mean_count = float(jnp.max(rates)) * years_to_seconds(observation_time)
    max_events = padded_event_capacity(mean_count, n_max_sigma)

    # Stage 3: the forward model, replaying the stage-1 hyperparameters.
    forward = partial(
        gwb_forward_model,
        source_model=source_model,
        merger_rate_fn=merger_rate_fn,
        generator=generator,
        observation_time=observation_time,
        batch_size=batch_size,
        max_events=max_events,
    )

    def model() -> None:
        theta: dict[str, Any] = dict(fixed)
        for name, prior in priors.items():
            theta[name] = numpyro.sample(name, prior)
        forward(theta)

    draws = Predictive(
        model,
        posterior_samples=sampled or None,
        num_samples=num_draws,
        return_sites=_RETURN_SITES,
    )(forward_key)

    return SpectralDensityDraws(
        frequencies=np.asarray(generator.frequencies),
        spectral_density=np.asarray(draws["spectral_density"]),
        n_events=np.asarray(draws["n_events"], dtype=np.int64),
        total_merger_rate=np.asarray(draws["total_merger_rate"], dtype=np.float64),
        hyperparameters={
            name: np.asarray(values, dtype=np.float64)
            for name, values in columns.items()
        },
    )
