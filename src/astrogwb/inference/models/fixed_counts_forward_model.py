r"""Fixed-count forward model, normalized by the observer-frame merger rate.

Each realization draws exactly ``num_events`` sources and records
``n_events``, ``total_merger_rate`` and ``spectral_density`` as deterministics.
The spectrum is :math:`A_{\rm inc}\mathcal{R}\sum_i P_i(f)/N`.
Missing inclination uses the same isotropic average of face-on power as the
Poisson model. Waveform power is reduced in batches without materializing
the full ``(F, num_events)`` array.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike

from astrogwb.gwb.spectral import inclination_averaging_factor
from astrogwb.populations import MergerRateFn, SourceFn
from astrogwb.waveform import PolarizationPowerGenerator

from ._forward import _require_luminosity_distance, _sum_polarization_power

__all__ = ["fixed_counts_forward_model"]


def fixed_counts_forward_model(
    params: Mapping[str, ArrayLike],
    *,
    source_model: SourceFn,
    merger_rate_fn: MergerRateFn,
    generator: PolarizationPowerGenerator,
    observation_time: float,
    batch_size: int,
    num_events: int,
) -> None:
    """Draw a positive static source count and reduce to a strain spectrum.

    ``num_events`` and ``batch_size`` are Python integers, static under JIT.
    ``params`` supplies the hyperparameters to both population callables;
    this model does not sample them. A physical observer-frame rate in
    mergers per second is required even though the count is fixed.

    ``observation_time`` remains positive, in years, as in the shared
    simulator interface. It cancels from this normalization: the spectrum
    uses the population rate times the sample mean power.

    Run :func:`~astrogwb.inference.validate_source_model` eagerly before
    generation to check the population against the waveform family.
    """
    if (
        isinstance(num_events, bool)
        or not isinstance(num_events, int)
        or num_events <= 0
    ):
        raise ValueError("num_events must be a positive static integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if observation_time <= 0.0:
        raise ValueError("observation_time must be positive")

    total_merger_rate = jnp.reshape(jnp.asarray(merger_rate_fn(params)), ())
    numpyro.deterministic("total_merger_rate", total_merger_rate)
    numpyro.deterministic("n_events", jnp.asarray(num_events))
    with numpyro.plate("events", num_events):
        sources = dict(source_model(params))
    _require_luminosity_distance(sources)
    power_sum = _sum_polarization_power(
        generator,
        sources,
        jnp.ones(num_events, dtype=bool),
        batch_size=batch_size,
    )
    numpyro.deterministic(
        "spectral_density",
        inclination_averaging_factor(sources)
        * total_merger_rate
        * power_sum
        / num_events,
    )
