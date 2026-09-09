"""The Madau-Dickinson (2017) rate shape as a redshift distribution factory."""

from __future__ import annotations

from collections.abc import Mapping

import jax
from jax.typing import ArrayLike

from astrogwb.distributions.rates import madau_dickinson_rate
from astrogwb.distributions.redshift.base import RedshiftDistribution


def _madau_dickinson_source_frame_distribution(
    redshift: ArrayLike, params: Mapping[str, ArrayLike]
) -> jax.Array:
    """Adapt the parameter mapping to the Madau-Dickinson rate function."""
    return madau_dickinson_rate(
        redshift,
        params["gamma"],
        params["kappa"],
        params["z_peak"],
    )


def madau_dickinson_redshift_distribution(
    *,
    params: Mapping[str, ArrayLike],
    minimum_redshift: float = 0.0,
    maximum_redshift: float = 10.0,
    n_grid: int = 1000,
    validate_args: bool | None = None,
) -> RedshiftDistribution:
    r"""Construct a redshift distribution for a Madau-Dickinson merger rate.

    ``params`` must carry ``gamma``, ``kappa`` and ``z_peak`` -- the arguments
    of :func:`~astrogwb.distributions.rates.madau_dickinson_rate` -- on top of
    the ``H0`` / ``Omega_m`` cosmology keys
    :class:`~astrogwb.distributions.redshift.base.RedshiftDistribution` reads.
    """
    return RedshiftDistribution(
        params=params,
        source_frame_distribution=_madau_dickinson_source_frame_distribution,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
        n_grid=n_grid,
        validate_args=validate_args,
    )


MadauDickinsonRedshiftDistribution = madau_dickinson_redshift_distribution
