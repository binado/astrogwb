r"""Exact Poisson-catalog forward model of the gravitational-wave spectrum.

The catalog contraction :func:`~astrogwb.gwb.spectral.spectral_density`
replaces a finite observation with its large-N mean,
:math:`S_h(f) = \mathcal{R}\,\langle P(f)\rangle`. This module draws the
catalog instead:

1. Evaluate the population's observer-frame merger rate :math:`\mathcal{R}`.
2. Draw :math:`N \sim \mathrm{Poisson}(\mathcal{R}\, T)`.
3. Draw ``max_events`` sources from the population under a NumPyro plate
   (a static size, so the model is a valid JAX pytree and ``jax.jit``
   target). Events with index :math:`\ge N` are masked out.
4. Generate polarization power over contiguous batches of ``batch_size``,
   reducing each chunk to ``(F,)`` before the next so the ``(F, N)`` array
   is never materialized, and form

   .. math::

       S_h(f) = \frac{1}{T}\sum_{i=1}^{\min(N,\, N_{\max})} P_i(f).

In expectation this recovers :math:`\mathcal{R}\,\langle P\rangle` whenever
``max_events`` upper-bounds the Poisson draw. There is no observation site:
the model is a simulator, consumed with :class:`~numpyro.infer.Predictive`
or ``jax.jit``.

Batching over the *event* axis is the memory-safe counterpart of
``lax.map(f, sources, batch_size=...)``. Mapping a per-source waveform
function would stack an ``(N, F)`` result -- the OOM the batching exists to
avoid. Each step therefore consumes a ``(batch_size,)`` catalog chunk,
calls the generator once, and returns the masked ``(F,)`` sum.
``max_events`` is padded to a multiple of ``batch_size`` so that path has
no remainder kernel.

Under :func:`jax.jit` the reduction is :func:`jax.lax.map`, which compiles
once per batch size. Eager execution (``Predictive`` without jit) uses a
Python loop over those same batches so generators with host-side control
flow -- Ripple's TaylorF2 path, which is what production catalogs use --
see concrete arrays. ``jax.lax.map`` always traces its body, and that
trace hits ``bool(jnp.any(...))`` inside ``gwmock_signal``.

``max_events`` sizes the source plate, so it is a Python integer (static
under JIT). ``N`` itself stays a traced Poisson draw. For several Predictive
draws the source sites have a fixed leading length and stack without
``return_sites``; omitting them is still cheaper. A JAX-native generator
(AnalyticInspiral) can wrap that Predictive in :func:`jax.jit`; Ripple
must not -- use the same call without ``jit``::

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
                max_events=1 << 18,
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
import jax.numpy as jnp
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
    population: Population, params: Mapping[str, ArrayLike], max_events: int
) -> dict[str, jax.Array]:
    """Draw ``max_events`` sources under a plate, hiding the plated rate site.

    The unplated rate is already published as ``total_merger_rate``. The
    population also declares that deterministic, and under a plate it would
    become an ``(N,)`` site of the same name.
    """
    with (
        numpyro.plate("events", max_events),
        handlers.block(hide=[_TOTAL_MERGER_RATE_SITE]),
    ):
        trace = handlers.trace(population).get_trace(params)
    return {name: jnp.asarray(trace[name]["value"]) for name in population.source_sites}


def _chunked_sources(
    sources: Mapping[str, jax.Array],
    mask: jax.Array,
    *,
    batch_size: int,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Reshape ``(N,)`` sources into ``(n_batches, batch_size)``, padding if needed."""
    n_events = array_dict_shape(sources)[0]
    pad = (batch_size - (n_events % batch_size)) % batch_size
    # Repeat a real source rather than padding with zeros: d_L = 0 and
    # M = 0 make the inspiral amplitude inf/NaN, and those values would
    # leak into the reduction even with a zero mask (NaN * 0 is NaN).
    padded = {
        name: jnp.pad(values, (0, pad), mode="edge") for name, values in sources.items()
    }
    padded_mask = jnp.pad(mask.astype(jnp.float64), (0, pad))
    n_batches = (n_events + pad) // batch_size
    chunked = {
        name: values.reshape((n_batches, batch_size)) for name, values in padded.items()
    }
    return chunked, padded_mask.reshape((n_batches, batch_size))


def _is_traced(value: jax.Array) -> bool:
    """True when ``value`` is a JAX tracer, so host-side Python control flow is unsafe."""
    return isinstance(value, jax.core.Tracer)


def _masked_batch_power(
    generator: PolarizationPowerGenerator,
    batch_sources: Mapping[str, jax.Array],
    batch_mask: jax.Array,
) -> jax.Array:
    """Call ``generator`` on one chunk and return the masked ``(F,)`` sum."""
    power = jnp.asarray(generator(batch_sources))
    n_chunk = batch_mask.shape[0]
    if power.ndim != 2 or power.shape[-1] != n_chunk:
        raise ValueError(
            "waveform generator must return frequency-first power of "
            f"shape (F, n_chunk); got {power.shape} for {n_chunk} sources"
        )
    return jnp.where(batch_mask > 0, power, 0.0).sum(axis=1)


def _sum_polarization_power(
    generator: PolarizationPowerGenerator,
    sources: Mapping[str, jax.Array],
    mask: jax.Array,
    *,
    batch_size: int,
) -> jax.Array:
    """Sum frequency-first polarization power over masked sources, chunk by chunk.

    Each step receives ``batch_size`` sources, calls ``generator`` once, and
    returns an ``(F,)`` masked sum, so peak waveform memory is
    ``(F, batch_size)`` rather than ``(F, N)``. Frequency count is taken from
    the generator output: Ripple only knows its grid after the first generate.
    """
    chunked, chunked_mask = _chunked_sources(sources, mask, batch_size=batch_size)

    def batch_power_sum(
        batch: tuple[dict[str, jax.Array], jax.Array],
    ) -> jax.Array:
        batch_sources, batch_mask = batch
        return _masked_batch_power(generator, batch_sources, batch_mask)

    if _is_traced(mask):
        batch_totals = jax.lax.map(batch_power_sum, (chunked, chunked_mask))
        return batch_totals.sum(axis=0)

    n_batches = next(iter(chunked.values())).shape[0]
    total = _masked_batch_power(
        generator,
        {name: values[0] for name, values in chunked.items()},
        chunked_mask[0],
    )
    for index in range(1, n_batches):
        batch_sources = {name: values[index] for name, values in chunked.items()}
        total = total + _masked_batch_power(
            generator, batch_sources, chunked_mask[index]
        )
    return total


def gwb_forward_model(
    params: Mapping[str, ArrayLike],
    *,
    population: Population,
    generator: PolarizationPowerGenerator,
    observation_time: float,
    batch_size: int,
    max_events: int,
    average_mode: AverageMode = "catalog_inclination",
) -> None:
    r"""Draw a Poisson catalog and reduce it to a strain spectral density.

    ``params`` is the hyperparameter dict the population already accepts --
    this model does not sample them. ``observation_time`` is in years, the
    same unit as :func:`~astrogwb.utils.years_to_seconds` and the analysis
    grid; the Poisson rate converts it against the population's mergers-per-
    second :math:`\mathcal{R}`.

    ``max_events`` and ``batch_size`` are Python integers and are static under
    JIT. The Poisson count is traced; sources with index ``>= n_events`` do
    not contribute. Counts above ``max_events`` are truncated.

    Registered sites:

    - ``n_events``, a ``sample`` from
      ``Poisson(total_merger_rate * observation_time_seconds)``;
    - ``total_merger_rate`` and ``spectral_density`` as deterministics.

    Source sites from ``population`` are sampled under the ``events`` plate
    of length ``max_events``. There is no ``spectral_density_obs`` site.

    ``average_mode`` is the same inclination convention as
    :func:`~astrogwb.gwb.spectral.spectral_density`. Face-on populations
    (inclination pinned at 0) pair with ``"analytic_inclination"``; a
    population that already samples inclination uses ``"catalog_inclination"``.
    """
    batch_size = _require_positive_int("batch_size", batch_size)
    max_events = _require_positive_int("max_events", max_events)
    observation_time = _require_positive_time(observation_time)
    observation_time_sec = years_to_seconds(observation_time)

    total_merger_rate = _population_total_merger_rate(population, params)
    numpyro.deterministic(_TOTAL_MERGER_RATE_SITE, total_merger_rate)
    n_events = numpyro.sample(
        "n_events", dist.Poisson(total_merger_rate * observation_time_sec)
    )

    sources = _draw_sources(population, params, max_events)
    mask = jnp.arange(max_events) < n_events
    power_sum = _sum_polarization_power(generator, sources, mask, batch_size=batch_size)

    factor = (
        INCLINATION_AVERAGE_TO_FACE_ON_RATIO
        if average_mode == "analytic_inclination"
        else 1.0
    )
    numpyro.deterministic("spectral_density", factor * power_sum / observation_time_sec)


__all__ = ["gwb_forward_model"]
