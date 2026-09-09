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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import distance_and_volume_grid
from astrogwb.distributions.interpolated import InterpolatedDistribution

type SourceFrameDistribution = Callable[[ArrayLike, Mapping[str, ArrayLike]], jax.Array]
"""Callable defining the unnormalized source-frame rate shape."""


class RedshiftDistribution(InterpolatedDistribution):
    r"""Redshift distribution corresponding to a merger-rate model.

    The source-frame rate shape :math:`\psi(z)` is supplied as a callable, and
    the distribution provides the cosmology, normalization, log density, and
    sampler. The distribution is normalized over
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
        ``source_frame_distribution`` reads. The keys match
        :data:`~astrogwb.paper.config.catalogs.MD_FIDUCIAL_NAMES` and the
        reference callback's, so one fiducials mapping feeds both with no
        translation layer.
    source_frame_distribution:
        Callable returning the unnormalized source-frame merger-rate shape
        :math:`\psi(z)`. It is retained as static pytree metadata.
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
    # `source_frame_distribution` is Python metadata, not numerical data: JAX
    # must restore the same callable when the distribution is flattened.
    pytree_aux_fields = ("_source_frame_distribution",)

    # `minimum_redshift` / `maximum_redshift` / `n_grid` were once declared as
    # aux fields but never assigned, so they flattened to `(None, None, None)`
    # and `tree_unflatten` faithfully restored three `None`s. They are now
    # properties over `self.x`, which is strictly better than assigning them:
    # aux data is hashed into the jit cache key, and the grid endpoints are
    # traced values under `jax.jit`, where a hashed Python float would either be
    # wrong or force a retrace per grid.
    pytree_data_fields = (
        "luminosity_distance_grid",
        "differential_comoving_volume_grid",
    )

    def __init__(
        self,
        *,
        params: Mapping[str, ArrayLike],
        source_frame_distribution: SourceFrameDistribution,
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
        self._source_frame_distribution = source_frame_distribution
        redshift_grid = jnp.linspace(minimum_redshift, maximum_redshift, n_grid)

        self.luminosity_distance_grid, self.differential_comoving_volume_grid = (
            distance_and_volume_grid(
                redshift_grid,
                hubble_constant=params["H0"],
                omega_m=params["Omega_m"],
            )
        )
        y = (
            self._source_frame_distribution(redshift_grid, params)
            / (1.0 + redshift_grid)
            * self.differential_comoving_volume_grid
        )
        super().__init__(redshift_grid, y, validate_args=validate_args)

    def source_frame_distribution(
        self, redshift: ArrayLike, params: Mapping[str, ArrayLike]
    ) -> jax.Array:
        r"""Forward to the supplied source-frame merger-rate shape :math:`\psi(z)`."""
        return self._source_frame_distribution(redshift, params)

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
