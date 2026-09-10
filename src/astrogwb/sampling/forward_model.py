r"""Exact Poisson-catalog forward model of the gravitational-wave spectrum.

The catalog contraction :func:`~astrogwb.gwb.spectral.spectral_density`
replaces a finite observation with its large-N mean,
:math:`S_h(f) = \mathcal{R}\,\langle P(f)\rangle`. This module draws a catalog
of a known size instead:

1. Evaluate the population's observer-frame merger rate :math:`\mathcal{R}`.
2. Observe :math:`N` as ``Poisson(\mathcal{R}\, T)`` at the plate size
   ``num_events``.
3. Draw ``num_events`` sources from the population under a NumPyro plate
   (a Python integer, so the model is a valid JAX pytree and ``jax.jit``
   target).
4. Generate polarization power over contiguous batches of ``batch_size``,
   reducing each chunk to ``(F,)`` before the next so the ``(F, N)`` array
   is never materialized, and form

   .. math::

       S_h(f) = \frac{1}{T}\sum_{i=1}^{N} P_i(f).

``num_events`` is the observed count, not the rate. The Poisson mean is
still :math:`\mathcal{R}\, T`; observing it at ``N`` puts :math:`p(N\mid
\mathcal{R}\, T)` in the joint without sampling a data-dependent plate size.
A traced Poisson draw cannot size a plate under :func:`jax.jit`. Each
distinct ``num_events`` is a different JIT specialization.

There is no ``spectral_density_obs`` site: the model is a simulator for a
catalog of size ``N``, consumed with :class:`~numpyro.infer.Predictive` or
``jax.jit``. To draw a random :math:`N` first, sample it in Python and pass
it as ``num_events``.

Batching over the *event* axis is the memory-safe counterpart of
``lax.map(f, sources, batch_size=...)``. Mapping a per-source waveform
function would stack an ``(N, F)`` result -- the OOM the batching exists to
avoid. Each step consumes a catalog chunk, calls the generator once, and
returns the ``(F,)`` sum. Full batches of ``batch_size`` are mapped; a
static remainder (``num_events % batch_size``) is a separate generate, so
dummy sources are never padded in.

Under :func:`jax.jit` -- and under ``Predictive`` with more than one
sample, which itself uses :func:`jax.lax.map` -- full batches use
:func:`jax.lax.map`. JAX-native generators (AnalyticInspiral) are called
directly. Ripple is not a tracing target (``gwmock_signal`` uses host-side
``bool`` checks), so those batches go through :func:`jax.pure_callback`.
Ripple's frequency grid is only known after the first generate, so the
callback needs a warmed generator. Eager single-trace execution uses a
Python loop over the same batches and does not need that warmup.

Warm a Ripple generator (one generate) before ``Predictive`` or
:func:`jax.jit` so the host callback can size its result::

    from functools import partial

    from numpyro.infer import Predictive

    from astrogwb.sampling import gwb_forward_model

    simulate = jax.jit(
        Predictive(
            partial(
                gwb_forward_model,
                population=population,
                generator=generator,
                observation_time=1.0,
                batch_size=1024,
                num_events=10_000,
            ),
            num_samples=1,
            return_sites=("spectral_density", "n_events", "total_merger_rate"),
        )
    )
    draws = simulate(jax.random.key(0), params)
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro import handlers

from astrogwb.constants import INCLINATION_AVERAGE_TO_FACE_ON_RATIO
from astrogwb.gwb.spectral import AverageMode
from astrogwb.populations import Population
from astrogwb.utils import array_dict_shape, years_to_seconds
from astrogwb.waveform import PolarizationPowerGenerator

_TOTAL_MERGER_RATE_SITE = "total_merger_rate"


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _require_positive_time(observation_time: float) -> float:
    time = float(observation_time)
    if not math.isfinite(time) or time <= 0.0:
        raise ValueError(
            f"observation_time must be a finite positive duration in years, "
            f"got {observation_time!r}"
        )
    return time


def _population_total_merger_rate(
    population: Population, params: Mapping[str, ArrayLike]
) -> jax.Array:
    """Observer-frame rate at ``params``, without consuming the outer RNG."""
    with handlers.block():
        trace = handlers.trace(handlers.seed(population, 0)).get_trace(params)
    if _TOTAL_MERGER_RATE_SITE not in trace:
        raise ValueError(
            "population declares no total_merger_rate site: params must carry "
            "the physical rate parameter (typically local_merger_rate) for a "
            "forward spectrum"
        )
    rate = jnp.asarray(trace[_TOTAL_MERGER_RATE_SITE]["value"])
    return jnp.reshape(rate, ())


def _draw_sources(
    population: Population, params: Mapping[str, ArrayLike], num_events: int
) -> dict[str, jax.Array]:
    """Draw ``num_events`` sources under a plate, hiding the plated rate site.

    The unplated rate is already published as ``total_merger_rate``. The
    population also declares that deterministic, and under a plate it would
    become an ``(N,)`` site of the same name. ``num_events`` must be positive:
    NumPyro plates reject size 0.
    """
    with (
        numpyro.plate("events", num_events),
        handlers.block(hide=[_TOTAL_MERGER_RATE_SITE]),
    ):
        trace = handlers.trace(population).get_trace(params)
    return {name: jnp.asarray(trace[name]["value"]) for name in population.source_sites}


def _is_traced(value: jax.Array) -> bool:
    """True when ``value`` is a JAX tracer, so host-side Python control flow is unsafe."""
    return isinstance(value, jax.core.Tracer)


def _generate_power(
    generator: PolarizationPowerGenerator,
    batch_sources: Mapping[str, jax.Array],
) -> jax.Array:
    """Evaluate the generator, using a host callback when tracing a non-JAX waveform."""
    prototype = next(iter(batch_sources.values()))
    if _is_traced(prototype) and not generator.jax_native:
        n_chunk = prototype.shape[0]
        result_shape = jax.ShapeDtypeStruct(
            (generator.frequencies.shape[0], n_chunk), jnp.float64
        )

        def host_generate(batch: dict[str, np.ndarray]) -> np.ndarray:
            return np.asarray(generator(batch), dtype=np.float64)

        return jax.pure_callback(host_generate, result_shape, dict(batch_sources))
    return jnp.asarray(generator(batch_sources))


def _batch_power_sum(
    generator: PolarizationPowerGenerator,
    batch_sources: Mapping[str, jax.Array],
) -> jax.Array:
    """Call ``generator`` on one chunk and return the ``(F,)`` sum."""
    power = _generate_power(generator, batch_sources)
    n_chunk = next(iter(batch_sources.values())).shape[0]
    if power.ndim != 2 or power.shape[-1] != n_chunk:
        raise ValueError(
            "waveform generator must return frequency-first power of "
            f"shape (F, n_chunk); got {power.shape} for {n_chunk} sources"
        )
    return power.sum(axis=1)


def _sum_full_batches(
    generator: PolarizationPowerGenerator,
    chunked: Mapping[str, jax.Array],
    *,
    traced: bool,
) -> jax.Array:
    """Reduce ``(n_full, batch_size)`` sources to an ``(F,)`` power sum."""
    if traced:
        return jax.lax.map(
            lambda batch: _batch_power_sum(generator, batch),
            dict(chunked),
        ).sum(axis=0)

    n_full = next(iter(chunked.values())).shape[0]
    total = _batch_power_sum(
        generator, {name: values[0] for name, values in chunked.items()}
    )
    for index in range(1, n_full):
        total = total + _batch_power_sum(
            generator, {name: values[index] for name, values in chunked.items()}
        )
    return total


def _sum_polarization_power(
    generator: PolarizationPowerGenerator,
    sources: Mapping[str, jax.Array],
    *,
    batch_size: int,
) -> jax.Array:
    """Sum frequency-first polarization power over sources, chunk by chunk.

    Full batches of ``batch_size`` are reduced together; a static remainder
    is a separate generate. Peak waveform memory is ``(F, batch_size)``
    rather than ``(F, N)``. Under a trace the host callback for a non-JAX
    generator needs ``generator.frequencies`` already cached.
    """
    n_events = array_dict_shape(sources)[0]
    if n_events == 0:
        return jnp.zeros(generator.frequencies.shape, dtype=jnp.float64)

    n_full, remainder = divmod(n_events, batch_size)
    traced = _is_traced(next(iter(sources.values())))
    if n_full:
        chunked = {
            name: values[: n_full * batch_size].reshape((n_full, batch_size))
            for name, values in sources.items()
        }
        total = _sum_full_batches(generator, chunked, traced=traced)
        if not remainder:
            return total
        tail = {name: values[n_full * batch_size :] for name, values in sources.items()}
        return total + _batch_power_sum(generator, tail)
    tail = {name: values[:remainder] for name, values in sources.items()}
    return _batch_power_sum(generator, tail)


def gwb_forward_model(
    params: Mapping[str, ArrayLike],
    *,
    population: Population,
    generator: PolarizationPowerGenerator,
    observation_time: float,
    batch_size: int,
    num_events: int,
    average_mode: AverageMode = "catalog_inclination",
) -> None:
    r"""Draw a catalog of size ``num_events`` and reduce it to a strain spectrum.

    ``params`` is the hyperparameter dict the population already accepts --
    this model does not sample them. ``observation_time`` is in years, the
    same unit as :func:`~astrogwb.utils.years_to_seconds` and the analysis
    grid; the Poisson rate converts it against the population's mergers-per-
    second :math:`\mathcal{R}`.

    ``num_events`` and ``batch_size`` are Python integers and are static under
    JIT. ``num_events`` is the plate dimension and the observed Poisson count
    (the event count, not the merger rate). A traced sample cannot size the
    plate. ``num_events = 0`` skips the plate (NumPyro requires a positive
    plate size) and yields a zero spectrum.

    Registered sites:

    - ``n_events``, an observed ``sample`` from
      ``Poisson(total_merger_rate * observation_time_seconds)``;
    - ``total_merger_rate`` and ``spectral_density`` as deterministics.

    Source sites from ``population`` are sampled under the ``events`` plate
    of length ``num_events`` when that length is positive. There is no
    ``spectral_density_obs`` site.

    ``average_mode`` is the same inclination convention as
    :func:`~astrogwb.gwb.spectral.spectral_density`. Face-on populations
    (inclination pinned at 0) pair with ``"analytic_inclination"``; a
    population that already samples inclination uses ``"catalog_inclination"``.
    """
    batch_size = _require_positive_int("batch_size", batch_size)
    num_events = _require_non_negative_int("num_events", num_events)
    observation_time = _require_positive_time(observation_time)
    observation_time_sec = years_to_seconds(observation_time)

    total_merger_rate = _population_total_merger_rate(population, params)
    numpyro.deterministic(_TOTAL_MERGER_RATE_SITE, total_merger_rate)
    numpyro.sample(
        "n_events",
        dist.Poisson(total_merger_rate * observation_time_sec),
        obs=num_events,
    )

    if num_events:
        sources = _draw_sources(population, params, num_events)
        power_sum = _sum_polarization_power(generator, sources, batch_size=batch_size)
    else:
        power_sum = jnp.zeros(generator.frequencies.shape, dtype=jnp.float64)

    factor = (
        INCLINATION_AVERAGE_TO_FACE_ON_RATIO
        if average_mode == "analytic_inclination"
        else 1.0
    )
    numpyro.deterministic("spectral_density", factor * power_sum / observation_time_sec)


__all__ = ["gwb_forward_model"]
