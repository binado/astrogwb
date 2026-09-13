r"""Exact Poisson-catalog forward model of the gravitational-wave spectrum.

The catalog contraction :func:`~astrogwb.gwb.spectral.spectral_density`
replaces a finite observation with its large-N mean,
:math:`S_h(f) = \mathcal{R}\,\langle P(f)\rangle` -- equivalently, this
module's exact sum with every importance weight pinned to one and the
empirical rate :math:`N/T` in place of :math:`\mathcal{R}`. This module draws
a catalog of a known size instead:

1. Evaluate the observer-frame merger rate :math:`\mathcal{R}` from
   ``merger_rate_fn``.
2. Observe :math:`N` as ``Poisson(\mathcal{R}\, T)`` at the plate size
   ``num_events``.
3. Draw ``num_events`` sources from ``source_model`` under a NumPyro plate
   (a Python integer, so the model is a valid JAX pytree and ``jax.jit``
   target). The source model returns that source dict, which is passed
   to the waveform generator.
4. Generate polarization power over contiguous batches of ``batch_size``
   with :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate_batch`,
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

The plated source draw already returns ``(N,)`` arrays, passed as a dict
to :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate_batch`.
Mapping per-source :meth:`~astrogwb.waveform.PolarizationPowerGenerator.generate`
would stack ``(N, F)`` -- the OOM the batching exists to avoid -- not because
of a vmap-inside-plate problem.

Full batches of ``batch_size`` are reduced with :func:`jax.lax.scan`, so the
compiled graph carries one batch body however many chunks there are; a static
remainder (``num_events % batch_size``) is a separate ``generate_batch``. Peak
waveform memory is ``(F, batch_size)`` per draw. The scan bounds the *waveform*
intermediate, not the NumPyro draw: the catalog of source parameters is still
materialized in full, so source storage remains ``O(num_events)``, and
:class:`~numpyro.infer.Predictive` adds a leading draw axis on top of that.

Shapes are static, values are not. ``num_events`` is a plate size, so each
distinct catalog size is its own JIT specialization, and the full-chunk and
remainder paths compile separately because their array shapes differ. Changing
hyperparameter *values* costs no compilation.

Both generators are JAX-native and valid :func:`jax.jit`,
:class:`~numpyro.infer.Predictive` and :func:`jax.vmap` targets.
Neither validates physical values during generation -- a traced array cannot
raise, and forcing the question would sync the host on every chunk. Run
:func:`validate_source_model` once before inference to check a population
against a generator's waveform family::

    from functools import partial

    from numpyro.infer import Predictive

    from astrogwb.sampling import gwb_forward_model, validate_source_model

    validate_source_model(
        params, source_model=source_model, generator=generator,
        rng_key=jax.random.key(0),
    )
    simulate = Predictive(
        partial(
            gwb_forward_model,
            source_model=source_model,
            merger_rate_fn=merger_rate_fn,
            generator=generator,
            observation_time=1.0,
            batch_size=1024,
            num_events=10_000,
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

from astrogwb.constants import INCLINATION_AVERAGE_TO_FACE_ON_RATIO
from astrogwb.gwb.spectral import AverageMode
from astrogwb.populations import MergerRateFn, SourceFn
from astrogwb.utils import array_dict_shape, years_to_seconds
from astrogwb.waveform import PolarizationPowerGenerator

#: The source output naming the effective distance governing waveform
#: amplitude. Required in every source model's returned mapping; see
#: :mod:`astrogwb.importance.spectral`, which checks the same key.
_LUMINOSITY_DISTANCE = "luminosity_distance"

#: The population-level rate site name, published outside the ``events``
#: plate the source draw runs inside.
_TOTAL_MERGER_RATE_SITE = "total_merger_rate"


def _require_luminosity_distance(sources: Mapping[str, jax.Array]) -> None:
    """Raise unless the source mapping names the distance scaling the waveform."""
    if _LUMINOSITY_DISTANCE not in sources:
        raise KeyError(
            f"source model must return {_LUMINOSITY_DISTANCE!r}: it is the "
            "distance governing waveform amplitude"
        )


def _batch_power_sum(
    generator: PolarizationPowerGenerator,
    batch_sources: Mapping[str, jax.Array],
) -> jax.Array:
    """Call ``generate_batch`` on one chunk and return the ``(F,)`` sum."""
    power = jnp.asarray(generator.generate_batch(batch_sources))
    n_chunk = next(iter(batch_sources.values())).shape[0]
    if power.ndim != 2 or power.shape[-1] != n_chunk:
        raise ValueError(
            "waveform generator must return frequency-first power of "
            f"shape (F, n_chunk); got {power.shape} for {n_chunk} sources"
        )
    return power.sum(axis=1)


def _sum_polarization_power(
    generator: PolarizationPowerGenerator,
    sources: Mapping[str, jax.Array],
    *,
    batch_size: int,
) -> jax.Array:
    """Sum frequency-first polarization power over sources, chunk by chunk.

    Full batches of ``batch_size`` are reduced with :func:`jax.lax.scan`, so
    the compiled body is one batch regardless of how many chunks there are; a
    static remainder is a separate ``generate_batch``. Peak waveform memory is
    ``(F, batch_size)`` rather than ``(F, N)``.

    The zero carry comes from ``generator.frequencies``, so an empty catalog
    returns zeros without tracing or calling a waveform kernel at all.
    """
    n_events = array_dict_shape(sources)[0]
    n_full, remainder = divmod(n_events, batch_size)
    total = jnp.zeros(np.shape(generator.frequencies)[0], dtype=jnp.float64)

    # A Python branch, not a traced one: n_full is static, and lax.scan over
    # zero chunks would still trace the waveform body it never runs.
    if n_full:
        chunks = {
            name: values[: n_full * batch_size].reshape(n_full, batch_size)
            for name, values in sources.items()
        }

        def accumulate(
            carry: jax.Array, chunk: Mapping[str, jax.Array]
        ) -> tuple[jax.Array, None]:
            return carry + _batch_power_sum(generator, chunk), None

        total, _ = jax.lax.scan(accumulate, total, chunks)
    if remainder:
        tail = {name: values[n_full * batch_size :] for name, values in sources.items()}
        total = total + _batch_power_sum(generator, tail)
    return total


def validate_source_model(
    params: Mapping[str, ArrayLike],
    *,
    source_model: SourceFn,
    generator: PolarizationPowerGenerator,
    rng_key: jax.Array | int,
) -> None:
    """Check a population against a generator's waveform family, eagerly.

    :func:`gwb_forward_model` cannot do this itself: under ``jax.jit`` the
    source arrays are tracers, and a tracer cannot drive a Python exception.
    So generation trusts its inputs, and this is how a caller earns that trust
    -- one eager draw of a single source, checked against the approximant,
    before handing the model to NUTS or :class:`~numpyro.infer.Predictive`.

    No waveform is evaluated, so this is cheap enough to call unconditionally.

    Raises:
        KeyError: If ``source_model`` does not return ``luminosity_distance``.
        ValueError: If a drawn value names a degree of freedom the generator's
            approximant does not carry -- a tidal population against an
            aligned-spin model, say, whose deformabilities would otherwise be
            silently ignored.
    """
    with numpyro.handlers.seed(rng_seed=rng_key), numpyro.plate("events", 1):
        sources = dict(source_model(params))
    _require_luminosity_distance(sources)
    generator.check_sources(sources)


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
) -> None:
    r"""Draw a catalog of size ``num_events`` and reduce it to a strain spectrum.

    ``params`` is the hyperparameter dict ``source_model`` and
    ``merger_rate_fn`` already accept -- this model does not sample them.
    Normally the callables :func:`~astrogwb.populations.build_source_model`
    and :func:`~astrogwb.populations.build_merger_rate_fn` return, built once
    per run and reused, since both hash by identity and a fresh, equal
    rebuild forces a jit recompile. ``observation_time`` is in years, the
    same unit as :func:`~astrogwb.utils.years_to_seconds` and the analysis
    grid; the Poisson rate converts it against ``merger_rate_fn``'s
    mergers-per-second :math:`\mathcal{R}`.

    ``num_events`` and ``batch_size`` are Python integers, static under JIT.
    ``num_events`` is the plate dimension and the observed
    Poisson count (the event count, not the merger rate). A traced sample
    cannot size the plate.

    Registered sites:

    - ``n_events``, an observed ``sample`` from
      ``Poisson(total_merger_rate * observation_time_seconds)``;
    - ``total_merger_rate`` and ``spectral_density`` as deterministics.

    Source sites from ``source_model`` are sampled under the ``events`` plate
    of length ``num_events``, published *after* ``total_merger_rate`` so the
    rate is always available even when ``num_events`` is zero. There is no
    ``spectral_density_obs`` site.

    ``average_mode`` is the same inclination convention as
    :func:`~astrogwb.gwb.spectral.spectral_density`. Face-on populations
    (inclination pinned at 0) pair with ``"analytic_inclination"``; a
    population that already samples inclination uses ``"catalog_inclination"``.

    Raises ``KeyError`` if ``source_model`` does not return
    ``luminosity_distance``: it is the distance governing waveform amplitude,
    and every registered source model must declare it.
    """
    observation_time_sec = years_to_seconds(observation_time)

    total_merger_rate = jnp.reshape(jnp.asarray(merger_rate_fn(params)), ())
    numpyro.deterministic(_TOTAL_MERGER_RATE_SITE, total_merger_rate)
    with numpyro.plate("events", num_events):
        sources = dict(source_model(params))
    _require_luminosity_distance(sources)
    numpyro.sample(
        "n_events",
        dist.Poisson(total_merger_rate * observation_time_sec),
        obs=num_events,
    )

    power_sum = _sum_polarization_power(generator, sources, batch_size=batch_size)

    factor = (
        INCLINATION_AVERAGE_TO_FACE_ON_RATIO
        if average_mode == "analytic_inclination"
        else 1.0
    )
    numpyro.deterministic("spectral_density", factor * power_sum / observation_time_sec)


__all__ = ["gwb_forward_model", "validate_source_model"]
