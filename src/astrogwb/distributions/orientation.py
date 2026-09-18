r"""Isotropic polar-angle distributions.

A direction uniform on the sphere has polar angle :math:`\theta` with
:math:`\cos\theta` uniform on :math:`[-1, 1]`. Sampling is
:math:`\theta = \arccos U` for :math:`U \sim \mathrm{Unif}(-1, 1)`, and the
density on :math:`[0, \pi]` is :math:`\frac12\sin\theta`. That is the law of
binary inclination :math:`\iota` for an isotropic source population.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions.util import validate_sample


class UniformCosThetaDistribution(dist.Distribution):
    r"""Polar angle of a direction uniform on the sphere.

    If :math:`U \sim \mathrm{Unif}(-1, 1)`, this is the law of
    :math:`\theta = \arccos U`. The density is :math:`\frac12\sin\theta` on
    :math:`[0, \pi]`; the CDF is :math:`(1 - \cos\theta)/2`; the inverse CDF is
    :math:`\arccos(1 - 2q)`.

    Parameters
    ----------
    validate_args:
        Forwarded to :class:`~numpyro.distributions.Distribution`.
    """

    arg_constraints = {}  # noqa: RUF012
    support = dist.constraints.interval(0.0, math.pi)

    def __init__(self, *, validate_args: bool | None = None) -> None:
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    def sample(
        self, key: jax.Array | None, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """Uniform cosine on ``[-1, 1]``, then the polar angle."""
        # `None` only exists to match the base-class signature; handlers
        # always pass a real key.
        assert key is not None
        cos_theta = jax.random.uniform(
            key, shape=sample_shape + self.batch_shape, minval=-1.0, maxval=1.0
        )
        return jnp.asarray(jnp.arccos(cos_theta))

    @validate_sample
    def log_prob(self, value: ArrayLike) -> jax.Array:
        return jnp.asarray(jnp.log(jnp.sin(value)) - jnp.log(2.0))

    def cdf(self, value: ArrayLike) -> jax.Array:
        return jnp.asarray((1.0 - jnp.cos(value)) / 2.0)

    def icdf(self, q: ArrayLike) -> jax.Array:
        return jnp.asarray(jnp.arccos(1.0 - 2.0 * q))
