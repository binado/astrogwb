"""A univariate density defined by a table of ``(x, y)`` samples.

The distribution owns an unnormalized density tabulated on a fixed 1D grid and
derives everything else -- the normalization, the CDF, the inverse CDF -- from
that one table with the trapezoid rule, so the density that
:meth:`InterpolatedDistribution.log_prob` reports and the density that
:meth:`InterpolatedDistribution.sample` draws from cannot fall out of step.

The grid is a *quadrature scheme*, not a set of distribution parameters. That
is why it is absent from ``arg_constraints`` and published through
:attr:`InterpolatedDistribution.support` instead; see the comment on
``pytree_data_fields``.
"""

from __future__ import annotations

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
    integral :attr:`norm`, is the density. Interpolating the *normalized*
    table with ``left=0.0, right=0.0`` is what makes this exact rather than
    approximate: the interpolant's own integral is then precisely the
    normalization, and values off the ends of the table get zero density --
    hence ``-inf`` log-density -- rather than an extrapolation of it.

    ``batch_shape`` is ``()``: ``x`` and ``y`` are a single table, and scalar
    parameters are the only thing this describes. Batch over hyperparameters
    with :func:`jax.vmap`, as the reference importance model does.

    ``x`` is *not* validated to be strictly increasing, nor ``y`` to be
    nonnegative. Neither is checkable inside a traced ``__init__``, the same
    situation :class:`~astrogwb.distributions.amplitude.AmplitudeConditional` documents for
    its explicit ``grid=``; the caller owns both invariants.

    Parameters
    ----------
    x:
        Strictly increasing abscissa, shape ``(n_grid,)``.
    y:
        Unnormalized density on ``x``, nonnegative, same shape.
    validate_args:
        Forwarded to :class:`~numpyro.distributions.Distribution`. With no
        ``arg_constraints`` and no ``@validate_sample`` on :meth:`log_prob` it
        has no observable effect here; it is accepted for API uniformity with
        every other NumPyro distribution.
    """

    # No `arg_constraints`. NumPyro validates each constrained argument's shape
    # against `batch_shape + event_shape` -- `()` here -- so declaring one for a
    # `(n_grid,)` grid would fail on construction. `x` and `y` are the
    # quadrature scheme, not distribution parameters; the grid bounds are
    # published through `support` instead.
    #
    # `norm`, `normalized_y` and `cdf_grid` are deliberately absent: they are
    # `lazy_property`, and `tree_flatten` reads `self.__dict__.get(name)`, so a
    # lazy field listed here would flatten to `None` before its first access and
    # to an array afterwards. The treedef would change under the object and every
    # `jax.jit` taking it as an argument would retrace.
    pytree_data_fields = ("x", "y")

    def __init__(
        self, x: ArrayLike, y: ArrayLike, *, validate_args: bool | None = None
    ) -> None:
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        """The closed interval the table spans.

        A ``dependent_property`` rather than a class attribute because the
        bounds are traced values read off ``x``. ``biject_to`` maps onto the
        *open* interval, so a latent site can never propose either endpoint.
        """
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

        The final divide is the one piece of hardening :meth:`icdf` needs. The
        cumulative sum reaches ``1.0`` only by a rounding coincidence -- it
        reduces in a different order than the ``sum`` behind :attr:`norm` --
        and dividing by the endpoint makes ``icdf(1.0) == x[-1]`` true by
        construction.

        Do *not* port :meth:`~astrogwb.distributions.amplitude.AmplitudeConditional.icdf`'s
        ``searchsorted``/``take_along_axis`` machinery on top of this. That
        exists because its CDF comes from ``exp(log_integrand - max)``, whose
        tails underflow to exact zeros and leave 0/0 plateaus -- a situation a
        physical density table cannot produce.
        """
        cdf = cumulative_trapezoid(self.normalized_y, x=self.x)
        return cdf / cdf[..., -1:]

    def log_prob(self, value: ArrayLike) -> jax.Array:
        """Log density, ``-inf`` off the table.

        Deliberately not ``@validate_sample``-decorated: importance weights
        built on this depend on getting ``-inf`` for an out-of-grid sample,
        not an exception.
        """
        pdf = jnp.interp(value, self.x, self.normalized_y, left=0.0, right=0.0)
        return jnp.log(pdf)

    def icdf(self, q: ArrayLike) -> jax.Array:
        """Inverse CDF by linear-in-CDF inversion on :attr:`cdf_grid`."""
        return jnp.interp(q, self.cdf_grid, self.x)

    def sample(
        self, key: jax.Array | None, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """Inverse-transform draw: one uniform per element, through :meth:`icdf`."""
        # `None` only exists to match the base-class signature; handlers
        # always pass a real key.
        assert key is not None
        return self.icdf(jax.random.uniform(key, sample_shape + self.batch_shape))
