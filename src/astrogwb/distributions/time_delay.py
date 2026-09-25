r"""Delay-time distributions between binary formation and merger.

A compact binary formed at lookback time :math:`t_f` merges at
:math:`t_f - \tau`, with the delay :math:`\tau` drawn from :math:`p(\tau)`.
Delays are in Gyr, the unit of :func:`astrogwb.cosmology.lookback_time`.

:class:`~astrogwb.distributions.redshift.time_delayed.TimeDelayedRedshiftDistribution`
integrates over the delay with quantile nodes, so any delay distribution it
takes must implement ``icdf``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.util import promote_shapes

#: Below this :math:`|\alpha + 1|` the log-uniform closed form is used, where
#: the general one cancels catastrophically.
_LOG_UNIFORM_TOLERANCE: float = 1e-6


class PowerLawTimeDelayDistribution(dist.Distribution):
    r"""Truncated power law :math:`p(\tau) \propto \tau^\alpha` on :math:`[\tau_{\min}, \tau_{\max}]`.

    With :math:`a = \alpha + 1`, the CDF is
    :math:`(\tau^a - \tau_{\min}^a) / (\tau_{\max}^a - \tau_{\min}^a)` and the
    inverse CDF is :math:`[\tau_{\min}^a + q(\tau_{\max}^a - \tau_{\min}^a)]^{1/a}`.
    At :math:`\alpha = -1` both reduce to the log-uniform law, which is
    evaluated in its own closed form. Both branches are always computed on
    safe inputs, so gradients in :math:`\alpha` stay finite across it.

    Parameters
    ----------
    alpha:
        Power-law slope. The canonical field-binary value is :math:`-1`.
    minimum_delay, maximum_delay:
        Support edges in Gyr, with ``0 < minimum_delay < maximum_delay``.
    validate_args:
        Forwarded to :class:`~numpyro.distributions.Distribution`.
    """

    arg_constraints = {  # noqa: RUF012
        "alpha": constraints.real,
        "minimum_delay": constraints.positive,
        "maximum_delay": constraints.positive,
    }
    reparametrized_params = ["alpha", "minimum_delay", "maximum_delay"]  # noqa: RUF012

    def __init__(
        self,
        alpha: ArrayLike,
        minimum_delay: ArrayLike,
        maximum_delay: ArrayLike,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.alpha, self.minimum_delay, self.maximum_delay = promote_shapes(
            jnp.asarray(alpha), jnp.asarray(minimum_delay), jnp.asarray(maximum_delay)
        )
        batch_shape = jnp.broadcast_shapes(
            jnp.shape(self.alpha),
            jnp.shape(self.minimum_delay),
            jnp.shape(self.maximum_delay),
        )
        super().__init__(
            batch_shape=batch_shape, event_shape=(), validate_args=validate_args
        )

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        return constraints.interval(self.minimum_delay, self.maximum_delay)

    def _exponent(self) -> tuple[jax.Array, jax.Array]:
        """``(is_log_uniform, a)`` with ``a = alpha + 1`` kept away from zero."""
        a = self.alpha + 1.0
        is_log_uniform = jnp.abs(a) < _LOG_UNIFORM_TOLERANCE
        return is_log_uniform, jnp.where(is_log_uniform, 1.0, a)

    def log_prob(
        self, value: ArrayLike, intermediates: list[Any] | None = None
    ) -> jax.Array:
        """Log density, ``-inf`` off the support.

        ``intermediates`` is accepted for signature compatibility with
        :class:`numpyro.distributions.Distribution` and ignored.
        """
        value = jnp.asarray(value)
        in_support = (value >= self.minimum_delay) & (value <= self.maximum_delay)
        safe_value = jnp.where(in_support, value, self.minimum_delay)
        is_log_uniform, a = self._exponent()
        log_norm = jnp.where(
            is_log_uniform,
            jnp.log(jnp.log(self.maximum_delay / self.minimum_delay)),
            jnp.log((self.maximum_delay**a - self.minimum_delay**a) / a),
        )
        log_density = self.alpha * jnp.log(safe_value) - log_norm
        return jnp.where(in_support, log_density, -jnp.inf)

    def cdf(self, value: ArrayLike) -> jax.Array:
        value = jnp.clip(jnp.asarray(value), self.minimum_delay, self.maximum_delay)
        is_log_uniform, a = self._exponent()
        return jnp.where(
            is_log_uniform,
            jnp.log(value / self.minimum_delay)
            / jnp.log(self.maximum_delay / self.minimum_delay),
            (value**a - self.minimum_delay**a)
            / (self.maximum_delay**a - self.minimum_delay**a),
        )

    def icdf(self, q: ArrayLike) -> jax.Array:
        q = jnp.asarray(q)
        is_log_uniform, a = self._exponent()
        return jnp.where(
            is_log_uniform,
            self.minimum_delay * (self.maximum_delay / self.minimum_delay) ** q,
            (
                self.minimum_delay**a
                + q * (self.maximum_delay**a - self.minimum_delay**a)
            )
            ** (1.0 / a),
        )

    def sample(
        self, key: jax.Array | None, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """Inverse-transform draw through :meth:`icdf`."""
        # `None` only exists to match the base-class signature; handlers
        # always pass a real key.
        assert key is not None
        return self.icdf(jax.random.uniform(key, sample_shape + self.batch_shape))
