r"""Delay-time distributions for mergers lagging their formation.

:class:`PowerLawDelayDistribution` is ``numpyro.distributions.DoublyTruncatedPowerLaw``
-- :math:`p(\tau) \propto \tau^{\alpha}` on :math:`[a, b]` -- with ``log_prob``,
``cdf`` and ``icdf`` rewritten so the slope can be a sampled hyperparameter.
Construction, support, pytree layout and ``sample`` (an ``icdf`` draw) are
inherited.

The parent special-cases :math:`\alpha = -1` exactly and otherwise evaluates
:math:`(b^{1+\alpha} - a^{1+\alpha}) / (1 + \alpha)` as written. Within
:math:`\sim 10^{-9}` of :math:`-1` that difference cancels, and the gradients of
its ``cdf`` and ``icdf`` in :math:`\alpha` reach :math:`\sim 10^{7}`: a NUTS step
landing there diverges, and :math:`\alpha = -1` is the canonical fiducial.

Here every expression is written in :math:`\beta = 1 + \alpha` through
:math:`h(x) = \operatorname{expm1}(x)/x` and :math:`g(y) = \operatorname{log1p}(y)/y`,
each switched to its Taylor series near zero. With :math:`\ell = \log(b/a)` and
:math:`s = \log(\tau/a)`,

.. math::

    F(\tau) = \frac{s\,h(\beta s)}{\ell\,h(\beta \ell)}, \qquad
    F^{-1}(u) = a \exp\!\big[g(y)\,u\,\ell\,h(\beta\ell)\big], \quad
    y = u\,\beta\ell\,h(\beta\ell),

which are smooth through :math:`\beta = 0` with no branch to differentiate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions.util import validate_sample

__all__ = ["PowerLawDelayDistribution"]

#: Below this magnitude `_expm1_ratio` and `_log1p_ratio` use their series. The
#: truncation error is O(x^3) ~ 1e-19 there, and the closed forms are accurate
#: to an ulp above it.
_SERIES_THRESHOLD = 1e-6


def _expm1_ratio(x: jax.Array) -> jax.Array:
    """``expm1(x) / x``, equal to 1 at 0, with a finite gradient everywhere."""
    small = jnp.abs(x) < _SERIES_THRESHOLD
    # Double `where`: the unused branch must stay finite, or its NaN gradient
    # leaks through the outer `where`.
    safe = jnp.where(small, 1.0, x)
    return jnp.where(small, 1.0 + x / 2.0 + x**2 / 6.0, jnp.expm1(safe) / safe)


def _log1p_ratio(y: jax.Array) -> jax.Array:
    """``log1p(y) / y``, equal to 1 at 0, with a finite gradient everywhere."""
    small = jnp.abs(y) < _SERIES_THRESHOLD
    safe = jnp.where(small, 1.0, y)
    return jnp.where(small, 1.0 - y / 2.0 + y**2 / 3.0, jnp.log1p(safe) / safe)


class PowerLawDelayDistribution(dist.DoublyTruncatedPowerLaw):
    r"""``DoublyTruncatedPowerLaw`` with a slope gradient smooth through :math:`-1`.

    Takes the parent's ``alpha``, ``low`` and ``high``: the index and the
    delay bounds, in the units the caller measures delays in (Gyr for
    :class:`~astrogwb.distributions.redshift.TimeDelayedRedshiftDistribution`).
    """

    def _shape(self) -> tuple[jax.Array, jax.Array]:
        """``(beta, log(high / low))``, the two numbers every formula needs."""
        return 1.0 + jnp.asarray(self.alpha), jnp.log(self.high / self.low)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> jax.Array:
        beta, log_range = self._shape()
        # Z = low^beta * log_range * h(beta * log_range).
        log_normalization = (
            beta * jnp.log(self.low)
            + jnp.log(log_range)
            + jnp.log(_expm1_ratio(beta * log_range))
        )
        return jnp.asarray(self.alpha) * jnp.log(value) - log_normalization

    def cdf(self, value: ArrayLike) -> jax.Array:
        beta, log_range = self._shape()
        log_ratio = jnp.log(jnp.asarray(value) / self.low)
        return (log_ratio * _expm1_ratio(beta * log_ratio)) / (
            log_range * _expm1_ratio(beta * log_range)
        )

    def icdf(self, q: ArrayLike) -> jax.Array:
        beta, log_range = self._shape()
        scaled_range = log_range * _expm1_ratio(beta * log_range)
        q = jnp.asarray(q)
        log_ratio = _log1p_ratio(q * beta * scaled_range) * q * scaled_range
        return self.low * jnp.exp(log_ratio)
