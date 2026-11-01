"""A univariate density defined by a table of ``(x, y)`` samples.

The normalization, CDF and inverse CDF all derive from one table by the
trapezoid rule, so ``log_prob`` and ``sample`` describe the same density.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.util import lazy_property

from astrogwb.utils import cumulative_trapezoid


class InterpolatedDistribution(dist.Distribution):
    r"""Density interpolated from an unnormalized table :math:`y(x)`.

    The linear interpolant of ``y`` on ``x``, divided by the trapezoidal
    integral :attr:`norm`, is the density.

    ``batch_shape`` is ``()``; batch over hyperparameters with :func:`jax.vmap`.

    Parameters
    ----------
    x:
        Strictly increasing abscissa, shape ``(n_grid,)``.
    y:
        Unnormalized density on ``x``, nonnegative, same shape.
    validate_args:
        Forwarded to :class:`~numpyro.distributions.Distribution`; it has no
        effect here.
    """

    # No `arg_constraints`: NumPyro would check the `(n_grid,)` table against
    # the `()` batch shape and fail.
    #
    # The lazy properties must stay out of this tuple: `tree_flatten` reads
    # `__dict__`, so they would flatten to `None` before first access and to an
    # array after, changing the treedef and retracing every jit.
    pytree_data_fields = ("x", "y")

    def __init__(
        self, x: ArrayLike, y: ArrayLike, *, validate_args: bool | None = None
    ) -> None:
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        """The closed interval the table spans; ``biject_to`` maps onto its interior."""
        return constraints.interval(self.x[0], self.x[-1])

    @lazy_property
    def norm(self) -> jax.Array:
        r"""Trapezoidal :math:`\int y\, dx` over the table."""
        return jnp.trapezoid(self.y, x=self.x)

    @lazy_property
    def normalized_y(self) -> jax.Array:
        """The table scaled to unit integral -- the density at the nodes."""
        return self.y / self.norm

    @lazy_property
    def cdf_grid(self) -> jax.Array:
        """Trapezoidal CDF at the nodes, exactly ``0.0`` to ``1.0``.

        Dividing by the endpoint guarantees ``icdf(1.0) == x[-1]``; the
        cumulative sum alone need not round to ``1.0``.
        """
        cdf = cumulative_trapezoid(self.normalized_y, x=self.x)
        return cdf / cdf[..., -1:]

    def log_prob(
        self, value: ArrayLike, intermediates: list[Any] | None = None
    ) -> jax.Array:
        """Log density, ``-inf`` off the table rather than raising.

        ``intermediates`` is unused. The operation order, ``log(interp(y) /
        norm)``, matches ``tests/reference_population.py`` bit for bit, which
        keeps a catalog that is its own proposal at exactly zero log-weight.
        """
        unnormalized_pdf = jnp.interp(value, self.x, self.y, left=0.0, right=0.0)
        return jnp.log(unnormalized_pdf / self.norm)

    def icdf(self, q: ArrayLike) -> jax.Array:
        """Inverse CDF by linear-in-CDF inversion on :attr:`cdf_grid`."""
        return jnp.interp(q, self.cdf_grid, self.x)

    def sample(
        self, key: jax.Array | None, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """Inverse-transform draw: one uniform per element, through :meth:`icdf`."""
        assert key is not None
        return self.icdf(jax.random.uniform(key, sample_shape + self.batch_shape))
