"""The Madau-Dickinson (2017) rate shape as redshift distribution factories."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.distributions.rates import madau_dickinson_rate
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.distributions.redshift.time_delayed import (
    TimeDelayedRedshiftDistribution,
)


def _madau_dickinson_source_frame_distribution(
    redshift: ArrayLike, params: Mapping[str, ArrayLike]
) -> jax.Array:
    """Adapt the parameter mapping to the Madau-Dickinson rate function."""
    return madau_dickinson_rate(
        redshift,
        params["gamma"],
        params["kappa"],
        params["z_peak"],
        params.get("local_merger_rate", 1.0),
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
    of :func:`~astrogwb.distributions.rates.madau_dickinson_rate` -- and may
    carry ``local_merger_rate`` for the absolute source-frame rate, on top of
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


def madau_dickinson_time_delayed_redshift_distribution(
    *,
    params: Mapping[str, ArrayLike],
    time_delay_distribution: dist.Distribution,
    n_delay_nodes: int = 200,
    maximum_formation_redshift: float = 20.0,
    minimum_redshift: float = 0.0,
    maximum_redshift: float = 10.0,
    n_grid: int = 1000,
    validate_args: bool | None = None,
) -> TimeDelayedRedshiftDistribution:
    r"""Mergers delayed from a Madau-Dickinson *formation* rate.

    ``params`` is as for :func:`madau_dickinson_redshift_distribution`;
    ``local_merger_rate`` stays the merger rate at :math:`z = 0`. See
    :class:`~astrogwb.distributions.redshift.time_delayed.TimeDelayedRedshiftDistribution`
    for the remaining arguments.
    """
    return TimeDelayedRedshiftDistribution(
        params=params,
        source_frame_distribution=_madau_dickinson_source_frame_distribution,
        time_delay_distribution=time_delay_distribution,
        n_delay_nodes=n_delay_nodes,
        maximum_formation_redshift=maximum_formation_redshift,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
        n_grid=n_grid,
        validate_args=validate_args,
    )
