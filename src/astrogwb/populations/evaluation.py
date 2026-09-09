"""Trace readers used by catalog consistency checks and inference."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro import handlers

from astrogwb.populations.base import Population, PopulationTrace

__all__ = [
    "LUMINOSITY_DISTANCE_SITE",
    "REDSHIFT_SITE",
    "TOTAL_MERGER_RATE_SITE",
    "PopulationTrace",
    "redshift_log_density",
]

#: The redshift site every population must declare. Its density can never be
#: excluded: redshift is the one source parameter the target and the proposal
#: are guaranteed to disagree on.
REDSHIFT_SITE = "redshift"

#: Deterministic distance governing waveform amplitude, in Mpc, shape ``(N,)``.
#: Includes modified GW propagation where the population declares it.
LUMINOSITY_DISTANCE_SITE = "luminosity_distance"

#: Deterministic observer-frame total merger rate, in mergers per second,
#: shape ``()``. A population declares it only when ``params`` carries the
#: physical rate parameters; a proposal density does not need one.
TOTAL_MERGER_RATE_SITE = "total_merger_rate"


def redshift_log_density(
    model: Population,
    params: Mapping[str, ArrayLike],
    redshift: ArrayLike,
) -> jax.Array:
    """The redshift site's log density at fixed probe values.

    Used as a drift fingerprint: a registered name pins the name, not the
    mathematics, so a catalog records this density at fixed probe redshifts and
    every load recomputes it. Reading the site's distribution rather than
    substituting values keeps the probe independent of which source columns a
    catalog happens to hold.
    """
    with handlers.block(), handlers.seed(rng_seed=0):
        trace = handlers.trace(model).get_trace(params)
    site = trace.get(REDSHIFT_SITE)
    if site is None or site["type"] != "sample":
        raise ValueError(
            f"population model declares no {REDSHIFT_SITE!r} sample site; every "
            "population must draw a redshift"
        )
    return jnp.asarray(site["fn"].log_prob(jnp.asarray(redshift)))
