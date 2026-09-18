"""Source-model adapters that add orientation degrees of freedom."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike

from astrogwb.distributions.orientation import UniformCosThetaDistribution
from astrogwb.populations.registry import SourceFn

__all__ = ["with_isotropic_inclination"]


def _source_model_with_isotropic_inclination(
    params: Mapping[str, ArrayLike],
    *,
    source_model: SourceFn,
) -> dict[str, jax.Array]:
    sources = dict(source_model(params))
    if "inclination" in sources:
        raise ValueError(
            "with_isotropic_inclination cannot wrap a source model that "
            "already returns 'inclination'"
        )
    inclination = numpyro.sample(
        "inclination",
        UniformCosThetaDistribution(validate_args=True),
    )
    return {**sources, "inclination": jnp.asarray(inclination)}


def with_isotropic_inclination(source_model: SourceFn) -> SourceFn:
    """Return ``source_model`` with an isotropic inclination site added.

    Binary inclination :math:`\\iota` is drawn from
    :class:`~astrogwb.distributions.orientation.UniformCosThetaDistribution`,
    the polar-angle law of a direction uniform on the sphere. The wrapped
    model returns the inner source mapping plus an ``inclination`` column, so
    waveform generators evaluate each source at its drawn orientation and
    :func:`~astrogwb.gwb.spectral.inclination_averaging_factor` does not apply
    the quadrupole analytic average.

    Inner source sites keep their names, so a shared PRNG key still replays
    the same masses, redshifts, and spins; only the new inclination site
    consumes an additional stream.
    """
    return partial(_source_model_with_isotropic_inclination, source_model=source_model)
