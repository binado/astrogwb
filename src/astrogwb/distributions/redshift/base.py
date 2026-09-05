r"""Redshift distributions induced by a source-frame merger-rate model.

A comoving source-frame merger-rate shape :math:`\psi(z)` induces a
detector-frame redshift density

.. math::

    p(z \mid \theta) \propto \frac{\psi(z)}{1 + z}\, \frac{\mathrm{d}V_c}{\mathrm{d}z},

with :math:`(1+z)^{-1}` the time dilation between the source and detector
frames. :class:`RedshiftDistribution` builds that unnormalized table on a
redshift grid and hands it to
:class:`~astrogwb.distributions.interpolated.InterpolatedDistribution`, which
owns the normalization, the log density, and the inverse-transform draw.

The class is the object-oriented counterpart of
:func:`~astrogwb.importance.models.bns_madau_dickinson_modified_propagation.compute_merger_rate_distance_and_logprob`
-- same formula, same grid convention, same units -- and
``tests/core/test_distributions.py`` pins the two together numerically. It does
not replace it: that function remains the single source of truth for the
importance-weighting path, whose proposal and target must provably come from
one implementation.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.distributions.distribution import DistributionMeta

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import distance_and_volume_grid
from astrogwb.distributions.interpolated import InterpolatedDistribution


class _AbstractDistributionMeta(abc.ABCMeta, DistributionMeta):
    """Metaclass merging ABC enforcement with NumPyro's distribution metaclass.

    ``class RedshiftDistribution(InterpolatedDistribution, abc.ABC)`` is a hard
    ``TypeError: metaclass conflict``: NumPyro's ``DistributionMeta`` and
    :class:`abc.ABCMeta` are unrelated subclasses of :class:`type`, so Python
    cannot pick a metaclass for the derived class. Deriving one from both is
    the standard resolution and costs nothing here -- ``ABCMeta.__new__`` still
    routes through ``type.__new__``, so ``Distribution.__init_subclass__``
    fires and every subclass is still registered as a JAX pytree node, and
    ``DistributionMeta.__init__`` still stamps ``__signature__``.

    ``DistributionMeta`` is imported from ``numpyro.distributions.distribution``
    because ``numpyro.distributions`` does not re-export it. That is the one
    genuinely private coupling in this module;
    ``test_redshift_distribution_cannot_be_instantiated`` and
    ``test_concrete_subclasses_are_still_registered_as_pytrees`` are what catch
    it going away.
    """


class RedshiftDistribution(
    InterpolatedDistribution, metaclass=_AbstractDistributionMeta
):
    r"""Redshift distribution corresponding to a merger-rate model.

    Abstract in :meth:`source_frame_distribution`: subclasses supply
    :math:`\psi(z)` and inherit the cosmology, the normalization, the log
    density, and the sampler. The distribution is normalized over
    ``[minimum_redshift, maximum_redshift]``.

    Note that the default ``minimum_redshift=0.0`` puts a zero-density point
    inside the support, since :math:`\mathrm{d}V_c/\mathrm{d}z` vanishes at
    :math:`z = 0`. That is the correct physics rather than a defect, and it is
    unreachable under NUTS: ``biject_to`` maps the real line onto the *open*
    interval, so neither endpoint is ever proposed.

    Parameters
    ----------
    params:
        Hyperparameters. Must include ``H0`` (in
        :math:`\mathrm{km\,s^{-1}\,Mpc^{-1}}`) and ``Omega_m``, plus whatever
        the subclass's :meth:`source_frame_distribution` reads. The keys match
        :data:`~astrogwb.paper.config.catalogs.MD_FIDUCIAL_NAMES` and the
        reference callback's, so one fiducials mapping feeds both with no
        translation layer.
    minimum_redshift, maximum_redshift, n_grid:
        The linear grid the cosmology integrals and the normalization run on.
        Read back afterwards off :attr:`redshift_grid` rather than stored.
    validate_args:
        Forwarded to :class:`~numpyro.distributions.Distribution`.
    """

    # `x` and `y` are contributed by `InterpolatedDistribution`:
    # `gather_pytree_data_fields` walks the MRO and unions each base's own
    # `pytree_data_fields`, so only the two new grids are named here.
    #
    # No `pytree_aux_fields`. `minimum_redshift` / `maximum_redshift` / `n_grid`
    # were once declared there but never assigned, so they flattened to
    # `(None, None, None)` and `tree_unflatten` faithfully restored three
    # `None`s. They are now properties over `self.x`, which is strictly better
    # than assigning them: aux data is hashed into the jit cache key, and the
    # grid endpoints are traced values under `jax.jit`, where a hashed Python
    # float would either be wrong or force a retrace per grid.
    pytree_data_fields = (
        "luminosity_distance_grid",
        "differential_comoving_volume_grid",
    )

    def __init__(
        self,
        *,
        params: Mapping[str, ArrayLike],
        minimum_redshift: float = 0.0,
        maximum_redshift: float = 10.0,
        n_grid: int = 1000,
        validate_args: bool | None = None,
    ) -> None:
        # A local, not `self.redshift_grid`: the grid becomes `self.x` in
        # `super().__init__`, and `redshift_grid` is a read-only alias for it.
        # Binding it here as well would put the same array in the pytree twice,
        # where a `tree_unflatten` with mismatched leaves could make the two
        # copies disagree.
        redshift_grid = jnp.linspace(minimum_redshift, maximum_redshift, n_grid)

        self.luminosity_distance_grid, self.differential_comoving_volume_grid = (
            distance_and_volume_grid(
                redshift_grid,
                hubble_constant=params["H0"],
                omega_m=params["Omega_m"],
            )
        )
        # The factor order matches `compute_merger_rate_distance_and_logprob`'s
        # `rate_shape_grid / (1 + z) * dvc_dz_grid` exactly, so `self.norm` is
        # bit-identical to its `integral_mpc3` and `total_merger_rate` agrees
        # with the reference to the last bit rather than to a tolerance.
        y = (
            self.source_frame_distribution(redshift_grid, params)
            / (1.0 + redshift_grid)
            * self.differential_comoving_volume_grid
        )
        super().__init__(redshift_grid, y, validate_args=validate_args)

    @abc.abstractmethod
    def source_frame_distribution(
        self, redshift: ArrayLike, params: Mapping[str, ArrayLike]
    ) -> jax.Array:
        r"""The comoving source-frame merger-rate shape :math:`\psi(z)`.

        Called from ``__init__`` before ``super().__init__``, so it must not
        touch any attribute the base class sets.
        """

    @property
    def redshift_grid(self) -> jax.Array:
        """The redshift grid -- an alias for the interpolant abscissa ``x``."""
        return self.x

    @property
    def minimum_redshift(self) -> jax.Array:
        """Lower grid edge, read back off the grid rather than stored."""
        return self.x[0]

    @property
    def maximum_redshift(self) -> jax.Array:
        """Upper grid edge, read back off the grid rather than stored."""
        return self.x[-1]

    @property
    def n_grid(self) -> int:
        """Number of grid nodes. Static even under tracing -- shapes always are."""
        return self.x.shape[-1]

    def total_merger_rate(self, local_merger_rate: ArrayLike) -> jax.Array:
        r"""Total merger rate, in mergers per second.

        .. math::

            \mathcal{R} = \mathcal{R}_0 \int \frac{\psi(z)}{1+z}
                \frac{\mathrm{d}V_c}{\mathrm{d}z}\,\mathrm{d}z

        :attr:`norm` is that integral in :math:`\mathrm{Mpc}^3` -- the density
        table carries :math:`\mathrm{d}V_c/\mathrm{d}z` in
        :math:`\mathrm{Mpc}^3` -- while ``local_merger_rate`` is in
        :math:`\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}`. So ``1e-9`` converts
        :math:`\mathrm{Mpc}^{-3}` to :math:`\mathrm{Gpc}^{-3}` and
        :data:`~astrogwb.constants.SECONDS_PER_YEAR` converts
        :math:`\mathrm{yr}^{-1}` to :math:`\mathrm{s}^{-1}`.
        :func:`~astrogwb.gwb.spectral.spectral_density` and every model in
        :mod:`astrogwb.sampling.models` take the rate in
        :math:`\mathrm{s}^{-1}`, which is why the conversion lives here rather
        than at each call site.

        A method taking ``local_merger_rate`` rather than a property, because
        the normalized redshift density does not depend on it: one distribution
        serves every draw of :math:`\mathcal{R}_0`, and the rate stays out of
        the pytree.
        """
        return 1e-9 * jnp.asarray(local_merger_rate) * self.norm / SECONDS_PER_YEAR

    def luminosity_distance(self, redshift: ArrayLike) -> jax.Array:
        """Luminosity distance in Mpc at redshift(s), clamped outside the grid."""
        return jnp.interp(jnp.asarray(redshift), self.x, self.luminosity_distance_grid)

    def differential_comoving_volume(self, redshift: ArrayLike) -> jax.Array:
        r"""All-sky :math:`\mathrm{d}V_c/\mathrm{d}z` in :math:`\mathrm{Mpc}^3`.

        Clamped to the grid endpoints outside ``[x[0], x[-1]]``.
        """
        return jnp.interp(
            jnp.asarray(redshift), self.x, self.differential_comoving_volume_grid
        )
