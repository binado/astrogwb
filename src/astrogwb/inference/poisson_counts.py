r"""Exact Poisson-catalog forward model of the gravitational-wave spectrum.

The catalog contraction :func:`~astrogwb.gwb.spectral.spectral_density`
replaces a finite observation with its large-N mean,
:math:`S_h(f) = \mathcal{R}\,\langle P(f)\rangle` -- equivalently, this
module's exact sum with every importance weight pinned to one and the
empirical rate :math:`N/T` in place of :math:`\mathcal{R}`. This module draws
a catalog into a static-capacity plate instead:

1. Evaluate the observer-frame merger rate :math:`\mathcal{R}` from
   the population.
2. Sample :math:`N` as ``Poisson(\mathcal{R}\, T)``.
3. Draw ``max_events`` sources from the population's model under a static NumPyro plate
   (a Python integer, so the model is a valid JAX pytree and ``jax.jit``
   target). The source model returns that source dict, which is passed to the
   waveform generator.
4. Generate polarization power over contiguous batches of ``chunk_size``
   with :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate_batch`,
   reducing each chunk to ``(F,)`` before the next so the ``(F, max_events)`` array
   is never materialized, and form

   .. math::

       S_h(f) = \frac{1}{T}\sum_{i=1}^{\min(N,\,\mathrm{max\_events})} P_i(f).

``max_events`` is a static capacity, not the observed count. The optional
``observed_num_events`` conditions the Poisson count when supplied; otherwise
``n_events`` is sampled. A traced Poisson draw only creates an elementwise
mask and never sizes a plate under :func:`jax.jit`. The source and waveform
arrays always have ``max_events`` slots, independent of the sampled count.

If the Poisson draw exceeds ``max_events``, all capacity slots are active and
the count site still reports the uncapped draw. Callers should choose
``max_events`` deep in the Poisson tail when that possibility is unacceptable.

The plated source draw already returns ``(max_events,)`` arrays, passed as a dict
to :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate_batch`.
Mapping per-source :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate`
would stack ``(max_events, F)`` -- the OOM the batching exists to avoid -- not because
of a vmap-inside-plate problem.

Full batches of ``chunk_size`` are reduced with :func:`jax.lax.scan`, so the
compiled graph carries one batch body however many chunks there are; a static
remainder (``max_events % chunk_size``) is a separate ``generate_batch``. Peak
waveform memory is ``(F, chunk_size)`` per draw. The scan bounds the *waveform*
intermediate, not the NumPyro draw: the catalog of source parameters is still
materialized in full, so source storage remains ``O(max_events)``, and
:class:`~numpyro.infer.Predictive` adds a leading draw axis on top of that.

Shapes are static, values are not. ``max_events`` is the plate size, so each
distinct capacity is its own JIT specialization, and the full-chunk and
remainder paths compile separately because their array shapes differ. Changing
hyperparameter *values* costs no compilation.

Both generators are JAX-native and valid :func:`jax.jit`,
:class:`~numpyro.infer.Predictive` and :func:`jax.vmap` targets.
Neither validates physical values during generation -- a traced array cannot
raise, and forcing the question would sync the host on every chunk::

    from functools import partial

    from numpyro.infer import Predictive

    from astrogwb.inference.poisson_counts import poisson_counts_forward_model

    simulate = Predictive(
        partial(
            poisson_counts_forward_model,
            population=population,
            generator=generator,
            observation_time=1.0,
            chunk_size=1024,
            max_events=10_000,
        ),
        num_samples=8,
        return_sites=("spectral_density", "n_events", "total_merger_rate"),
    )
    draws = simulate(jax.random.key(0), params)
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.populations import Population, joint_model
from astrogwb.utils import array_dict_shape, years_to_seconds
from astrogwb.waveform import PolarizationPowerGenerator

_TOTAL_MERGER_RATE_SITE = "total_merger_rate"


def _batch_power_sum(
    generator: PolarizationPowerGenerator,
    batch_sources: Mapping[str, jax.Array],
    event_mask: jax.Array,
) -> jax.Array:
    """Call ``generate_batch`` on one chunk and return its masked ``(F,)`` sum."""
    power = jnp.asarray(generator.generate_batch(batch_sources))
    n_chunk = next(iter(batch_sources.values())).shape[0]
    if power.ndim != 2 or power.shape[-1] != n_chunk:
        raise ValueError(
            "waveform generator must return frequency-first power of "
            f"shape (F, n_chunk); got {power.shape} for {n_chunk} sources"
        )
    return (power * event_mask).sum(axis=1)


def _sum_polarization_power(
    generator: PolarizationPowerGenerator,
    sources: Mapping[str, jax.Array],
    event_mask: ArrayLike,
    *,
    chunk_size: int,
) -> jax.Array:
    """Sum frequency-first polarization power over sources, chunk by chunk.

    Full batches of ``chunk_size`` are reduced with :func:`jax.lax.scan`, so
    the compiled body is one batch regardless of how many chunks there are; a
    static remainder is a separate ``generate_batch``. Peak waveform memory is
    ``(F, chunk_size)`` rather than ``(F, max_events)``.

    ``event_mask`` is sliced alongside each source batch, so inactive capacity
    slots do not contribute to the sum. The zero carry comes from
    ``generator.frequencies``, so an empty catalog returns zeros without
    tracing or calling a waveform kernel at all.
    """
    n_events = array_dict_shape(sources)[0]
    event_mask = jnp.asarray(event_mask)
    n_full, remainder = divmod(n_events, chunk_size)
    total = jnp.zeros(np.shape(generator.frequencies)[0], dtype=jnp.float64)

    # A Python branch, not a traced one: n_full is static, and lax.scan over
    # zero chunks would still trace the waveform body it never runs.
    if n_full:
        chunks = {
            name: values[: n_full * chunk_size].reshape(n_full, chunk_size)
            for name, values in sources.items()
        }

        chunk_masks = event_mask[: n_full * chunk_size].reshape(n_full, chunk_size)

        def accumulate(
            carry: jax.Array,
            scan_values: tuple[Mapping[str, jax.Array], jax.Array],
        ) -> tuple[jax.Array, None]:
            chunk, chunk_mask = scan_values
            return carry + _batch_power_sum(generator, chunk, chunk_mask), None

        total, _ = jax.lax.scan(
            accumulate,
            total,
            (chunks, chunk_masks),
        )
    if remainder:
        tail = {name: values[n_full * chunk_size :] for name, values in sources.items()}
        tail_mask = event_mask[n_full * chunk_size :]
        total = total + _batch_power_sum(generator, tail, tail_mask)
    return total


def poisson_counts_forward_model(
    params: Mapping[str, ArrayLike],
    *,
    population: Population,
    generator: PolarizationPowerGenerator,
    observation_time: float,
    chunk_size: int,
    max_events: int,
    observed_num_events: ArrayLike | None = None,
) -> None:
    r"""Draw up to ``max_events`` sources and reduce them to a strain spectrum.

    ``params`` is the hyperparameter dict ``population`` accepts -- this model
    does not sample them. ``population`` is built once per run and reused,
    since it hashes by identity and a fresh, equal rebuild forces a jit
    recompile. A proposal population has no meaningful Poisson mean and must
    not drive this model. ``observation_time`` is in years, the same unit as
    :func:`~astrogwb.utils.years_to_seconds` and the analysis grid; the Poisson
    rate converts it against the population's mergers-per-second
    :math:`\mathcal{R}`.

    ``max_events`` and ``chunk_size`` are Python integers, static under JIT.
    ``max_events`` is the plate dimension and capacity. ``n_events`` is an
    unobserved Poisson draw by default, or is conditioned on
    ``observed_num_events`` when supplied. A traced count only creates an
    elementwise source mask and cannot size the plate.

    Registered sites:

    - ``n_events``, a ``sample`` from
      ``Poisson(total_merger_rate * observation_time_seconds)``;
    - ``total_merger_rate`` and ``spectral_density`` as deterministics.

    Source sites from the population's model are sampled under the ``events`` plate
    of length ``max_events``, with inactive slots masked from their log
    density. The source arrays and waveform reduction retain that static
    capacity. If ``n_events > max_events``, the capacity is silently capped:
    all slots contribute, while the count site retains the actual draw. There
    is no ``spectral_density_obs`` site.

    Raises ``KeyError`` if the population's model does not return
    ``luminosity_distance``: it is the distance governing waveform amplitude,
    and every registered source model must declare it.
    """
    observation_time_sec = years_to_seconds(observation_time)

    redshift_distribution, source_model = population(params)
    model = joint_model(redshift_distribution, source_model)
    total_merger_rate = jnp.reshape(redshift_distribution.total_merger_rate(), ())
    numpyro.deterministic(_TOTAL_MERGER_RATE_SITE, total_merger_rate)
    n_events = numpyro.sample(
        "n_events",
        dist.Poisson(total_merger_rate * observation_time_sec),
        obs=observed_num_events,
    )
    event_mask = jnp.arange(max_events) < n_events
    with numpyro.plate("events", max_events), numpyro.handlers.mask(mask=event_mask):
        sources = dict(model())
    power_sum = _sum_polarization_power(
        generator,
        sources,
        event_mask,
        chunk_size=chunk_size,
    )

    numpyro.deterministic(
        "spectral_density",
        power_sum / observation_time_sec,
    )


__all__ = ["poisson_counts_forward_model"]
