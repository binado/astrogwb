r"""Component-mass distributions used by the BNS population models.

:class:`MaxOfTwoNormals` is the primary-mass marginal of two i.i.d. Gaussians
after ordering. Paired with a ``TruncatedNormal(..., high=m1)`` secondary it
gives the joint ``2\,\mathcal{N}(m_1)\,\mathcal{N}(m_2)`` on ``m_1 \ge m_2``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import ndtri
from jax.typing import ArrayLike
from numpyro.distributions.util import promote_shapes, validate_sample


class MaxOfTwoNormals(dist.Distribution):
    r"""Larger of two i.i.d. normals.

    If :math:`X, Y \sim \mathcal{N}(\mu, \sigma^2)` independently, this is the
    law of :math:`\max(X, Y)`. The density is ``2 \phi(x) \Phi(x)`` in standard
    units; the CDF is :math:`[\Phi((x-\mu)/\sigma)]^2`; inverse-transform sampling
    is :math:`\mu + \sigma\,\Phi^{-1}(\sqrt{u})`.

    Parameters
    ----------
    loc:
        Mean :math:`\mu` of each component.
    scale:
        Standard deviation :math:`\sigma` of each component.
    validate_args:
        Forwarded to :class:`~numpyro.distributions.Distribution`.
    """

    arg_constraints = {  # noqa: RUF012
        "loc": dist.constraints.real,
        "scale": dist.constraints.positive,
    }
    support = dist.constraints.real
    reparametrized_params = ["loc", "scale"]  # noqa: RUF012

    def __init__(
        self,
        loc: ArrayLike = 0.0,
        scale: ArrayLike = 1.0,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.scale = promote_shapes(loc, scale)
        batch_shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(
        self, key: jax.Array | None, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """Inverse-transform draw: one uniform per element, through :meth:`icdf`."""
        # `None` only exists to match the base-class signature; handlers
        # always pass a real key.
        assert key is not None
        tiny = jnp.finfo(jnp.result_type(float)).tiny
        u = jax.random.uniform(
            key,
            shape=sample_shape + self.batch_shape,
            minval=tiny,
            maxval=1.0 - tiny,
        )
        return jnp.asarray(self.icdf(u))

    @validate_sample
    def log_prob(self, value: ArrayLike) -> jax.Array:
        component = dist.Normal(self.loc, self.scale)
        return jnp.asarray(
            jnp.log(2.0) + component.log_prob(value) + component.log_cdf(value)
        )

    def cdf(self, value: ArrayLike) -> jax.Array:
        return jnp.asarray(dist.Normal(self.loc, self.scale).cdf(value) ** 2)

    def icdf(self, q: ArrayLike) -> jax.Array:
        return jnp.asarray(self.loc + self.scale * ndtri(jnp.sqrt(q)))
