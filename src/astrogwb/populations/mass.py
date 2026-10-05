r"""Registered component-mass laws a population is composed from.

A mass model is a registered factory returning a :data:`MassFn`: a callable
that declares the two stored mass sites, ``source_frame_mass_1`` and
``source_frame_mass_2``, and returns them. The pair is one unit rather than one
model per component because the second mass is conditioned on the first: the
components are an ordered pair, the first mass being the larger.

Both sites take part in importance weighting. Two laws are registered:

- ``ordered_uniform``: i.i.d. uniforms on ``[minimum_mass, minimum_mass +
  mass_width]``, then ordered; the hyperparameters are ``minimum_mass`` and
  ``mass_width``. The joint density is the constant ``2 / width**2`` on the
  ordered triangle. That triangle is compact, so a NUTS step that moves the
  edges can send catalog samples outside the support and drop their importance
  weights to zero.
- ``ordered_gaussian``: i.i.d. :math:`\mathcal{N}(\mu, \sigma^2)`, then
  ordered; the hyperparameters are ``mass_mean`` and ``mass_sigma``. The joint
  density is ``2\,\mathcal{N}(m_1)\,\mathcal{N}(m_2)`` on the half-plane
  ``m_1 \ge m_2``, with no compact support, so moving :math:`(\mu, \sigma)`
  never sends an importance weight to zero -- the property NUTS needs.
  Galactic BNS masses motivate the shape (a Gaussian around
  :math:`1.33\,M_\odot` with width :math:`\sim 0.09\,M_\odot`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.distributions.mass import MaxOfTwoNormalsDistribution
from astrogwb.populations.registry import ComponentRegistry

__all__ = [
    "MassFn",
    "build_mass_model",
    "known_mass_models",
    "register_mass_model",
]

#: Declares ``source_frame_mass_1`` and ``source_frame_mass_2`` and returns them.
type MassFn = Callable[[Mapping[str, ArrayLike]], tuple[jax.Array, jax.Array]]

_MASS_MODELS = ComponentRegistry("mass model")
register_mass_model = _MASS_MODELS.register
build_mass_model = _MASS_MODELS.build


def known_mass_models() -> tuple[str, ...]:
    """Every registered mass model name, in sorted order."""
    return _MASS_MODELS.names()


@register_mass_model("ordered_uniform")
def _ordered_uniform() -> MassFn:
    """Ordered pair of i.i.d. uniforms on ``[minimum_mass, minimum_mass + width]``."""

    def declare(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, jax.Array]:
        minimum_mass: jax.Array = jnp.asarray(params["minimum_mass"])
        mass_width: jax.Array = jnp.asarray(params["mass_width"])
        # For two ordered iid uniforms, Beta(2, 1) is the primary mass marginal.
        mass_1 = numpyro.sample(
            "source_frame_mass_1",
            dist.TransformedDistribution(
                dist.Beta(2.0, 1.0, validate_args=True),
                dist.transforms.AffineTransform(minimum_mass, mass_width),
                validate_args=True,
            ),
        )
        mass_2 = numpyro.sample(
            "source_frame_mass_2",
            dist.Uniform(minimum_mass, mass_1, validate_args=True),
        )
        return jnp.asarray(mass_1), jnp.asarray(mass_2)

    return declare


@register_mass_model("ordered_gaussian")
def _ordered_gaussian() -> MassFn:
    """Ordered pair of i.i.d. ``Normal(mass_mean, mass_sigma)`` components."""

    def declare(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, jax.Array]:
        mass_mean: jax.Array = jnp.asarray(params["mass_mean"])
        mass_sigma: jax.Array = jnp.asarray(params["mass_sigma"])
        mass_1 = numpyro.sample(
            "source_frame_mass_1",
            MaxOfTwoNormalsDistribution(mass_mean, mass_sigma, validate_args=True),
        )
        mass_2 = numpyro.sample(
            "source_frame_mass_2",
            dist.TruncatedNormal(
                mass_mean, mass_sigma, high=mass_1, validate_args=True
            ),
        )
        return jnp.asarray(mass_1), jnp.asarray(mass_2)

    return declare
