"""NumPyro handlers that add source orientation degrees of freedom."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro.primitives import Messenger

from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.populations.registry import SourceFn

__all__ = ["IsotropicInclination"]


class IsotropicInclination(Messenger):
    """Add an isotropic inclination sample site and output column.

    Binary inclination :math:`\\iota` is drawn from
    :class:`~astrogwb.distributions.orientation.UniformCosineDistribution`,
    the polar-angle law of a direction uniform on the sphere. The wrapped
    source model returns its original mapping plus an ``inclination`` column,
    so waveform generators evaluate each source at its drawn orientation and
    :func:`~astrogwb.gwb.spectral.inclination_averaging_factor` does not apply
    the quadrupole analytic average.

    The source model and the new sample site execute inside this Messenger's
    handler scope, so outer handlers such as ``seed``, ``trace``, and
    ``condition`` apply to both.
    """

    def __init__(self, source_model: SourceFn) -> None:
        super().__init__(source_model)
        self._source_model = source_model

    def __call__(self, params: Mapping[str, ArrayLike]) -> dict[str, jax.Array]:
        with self:
            sources = dict(self._source_model(params))
            if "inclination" in sources:
                raise ValueError(
                    "IsotropicInclination cannot wrap a source model that "
                    "already returns 'inclination'"
                )
            inclination = numpyro.sample(
                "inclination",
                UniformCosineDistribution(validate_args=True),
            )
            return {**sources, "inclination": jnp.asarray(inclination)}
