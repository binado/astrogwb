r"""Exact Poisson-catalog forward model of the gravitational-wave spectrum."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.constants import INCLINATION_AVERAGE_TO_FACE_ON_RATIO
from astrogwb.gwb.spectral import AverageMode
from astrogwb.populations import MergerRateFn, SourceFn
from astrogwb.utils import array_dict_shape, years_to_seconds
from astrogwb.waveform import PolarizationPowerGenerator

_LUMINOSITY_DISTANCE = "luminosity_distance"
_TOTAL_MERGER_RATE_SITE = "total_merger_rate"


def _require_luminosity_distance(sources: Mapping[str, jax.Array]) -> None:
    if _LUMINOSITY_DISTANCE not in sources:
        raise KeyError(
            f"source model must return {_LUMINOSITY_DISTANCE!r}: it is the "
            "distance governing waveform amplitude"
        )


def _batch_power_sum(
    generator: PolarizationPowerGenerator,
    batch_sources: Mapping[str, jax.Array],
    *,
    active: jax.Array | None = None,
) -> jax.Array:
    """Call ``generate_batch`` on one chunk and return the masked ``(F,)`` sum."""
    power = jnp.asarray(generator.generate_batch(batch_sources))
    n_chunk = next(iter(batch_sources.values())).shape[0]
    if power.ndim != 2 or power.shape[-1] != n_chunk:
        raise ValueError(
            "waveform generator must return frequency-first power of "
            f"shape (F, n_chunk); got {power.shape} for {n_chunk} sources"
        )
    if active is None:
        return power.sum(axis=1)
    return jnp.where(jnp.asarray(active)[jnp.newaxis, :], power, 0.0).sum(axis=1)


def _sum_polarization_power(
    generator: PolarizationPowerGenerator,
    sources: Mapping[str, jax.Array],
    *,
    batch_size: int,
    active: jax.Array | None = None,
) -> jax.Array:
    """Sum polarization power over sources in static chunks."""
    n_events = array_dict_shape(sources)[0]
    n_full, remainder = divmod(n_events, batch_size)

    if n_events == 0:
        return jnp.zeros(np.shape(generator.frequencies)[0], dtype=jnp.float64)

    first_chunk = min(batch_size, n_events)
    first_sources = {
        name: values[:first_chunk] for name, values in sources.items()
    }
    first_active = None if active is None else jnp.asarray(active)[:first_chunk]
    total = _batch_power_sum(generator, first_sources, active=first_active)

    if n_full > 1:
        chunks = {
            name: values[first_chunk : n_full * batch_size].reshape(
                n_full - 1, batch_size
            )
            for name, values in sources.items()
        }
        active_chunks = None
        if active is not None:
            active_chunks = jnp.asarray(active)[first_chunk : n_full * batch_size].reshape(
                n_full - 1, batch_size
            )

        def accumulate(carry: jax.Array, chunk: Mapping[str, jax.Array]) -> tuple[jax.Array, None]:
            return carry + _batch_power_sum(generator, chunk), None

        def accumulate_masked(
            carry: jax.Array, chunk: tuple[Mapping[str, jax.Array], jax.Array]
        ) -> tuple[jax.Array, None]:
            chunk_sources, chunk_active = chunk
            return carry + _batch_power_sum(
                generator, chunk_sources, active=chunk_active
            ), None

        if active_chunks is None:
            total, _ = jax.lax.scan(accumulate, total, chunks)
        else:
            total, _ = jax.lax.scan(accumulate_masked, total, (chunks, active_chunks))

    if remainder:
        start = n_full * batch_size
        tail = {name: values[start:] for name, values in sources.items()}
        tail_active = None if active is None else jnp.asarray(active)[start:]
        total = total + _batch_power_sum(generator, tail, active=tail_active)
    return total


def validate_source_model(
    params: Mapping[str, ArrayLike],
    *,
    source_model: SourceFn,
    generator: PolarizationPowerGenerator,
    rng_key: jax.Array | int,
) -> None:
    """Eagerly draw one source and validate that generation is possible."""
    with numpyro.handlers.seed(rng_seed=rng_key), numpyro.plate("events", 1):
        sources = dict(source_model(params))
    _require_luminosity_distance(sources)
    generator.generate_batch(sources)


def gwb_forward_model(
    params: Mapping[str, ArrayLike],
    *,
    source_model: SourceFn,
    merger_rate_fn: MergerRateFn,
    generator: PolarizationPowerGenerator,
    observation_time: float,
    batch_size: int,
    num_events: int,
    average_mode: AverageMode = "catalog_inclination",
    n_active: ArrayLike | None = None,
) -> None:
    r"""Draw a padded catalog and reduce it to a strain spectrum."""
    observation_time_sec = years_to_seconds(observation_time)

    total_merger_rate = jnp.reshape(jnp.asarray(merger_rate_fn(params)), ())
    numpyro.deterministic(_TOTAL_MERGER_RATE_SITE, total_merger_rate)

    active = None
    observed_n: int | jax.Array = num_events
    if n_active is not None:
        observed_n = jnp.reshape(jnp.asarray(n_active), ())
        active = jnp.arange(num_events) < observed_n

    with numpyro.plate("events", num_events):
        if active is None:
            sources = dict(source_model(params))
        else:
            with numpyro.handlers.mask(mask=active):
                sources = dict(source_model(params))

    _require_luminosity_distance(sources)
    numpyro.sample(
        "n_events",
        dist.Poisson(total_merger_rate * observation_time_sec),
        obs=observed_n,
    )

    power_sum = _sum_polarization_power(
        generator,
        sources,
        batch_size=batch_size,
        active=active,
    )

    factor = (
        INCLINATION_AVERAGE_TO_FACE_ON_RATIO
        if average_mode == "analytic_inclination"
        else 1.0
    )
    numpyro.deterministic("spectral_density", factor * power_sum / observation_time_sec)


__all__ = ["gwb_forward_model", "validate_source_model"]
