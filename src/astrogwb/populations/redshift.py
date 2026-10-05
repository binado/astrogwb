r"""Registered redshift laws a population is composed from.

A redshift model is a registered factory: it takes its construction kwargs (the
redshift window and grid, plus whatever else that law needs) and returns a
:class:`RedshiftModel`, whose ``law`` is a :data:`RedshiftFn` -- a callable from
hyperparameters to a :class:`RedshiftLaw`. The law is rebuilt on every call
because the hyperparameters (``H0``, ``z_peak``, ``delay_slope``, ...) are
traced values, so the distribution cannot exist before they do.

What the population needs from a redshift law is exactly what a
:class:`~astrogwb.distributions.redshift.RedshiftDistribution` offers: a
density to draw ``redshift`` from, the luminosity distance on the cosmology the
density was normalized with, and -- for a physical law -- its total merger rate,
which is that same normalization. Reading all three off one object is what stops
a rate being paired with a density that is not its own.

A guard mixture is a sampling density rather than a physical law: the
Madau-Dickinson total rate normalizes the Madau-Dickinson density, not a mixture
of it with a uniform component. It is registered with
``has_merger_rate=False``, a static flag so that "this population has no rate"
is known when the population is built, not when it is first executed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.cosmology import lookback_time
from astrogwb.distributions.delay import PowerLawDelayDistribution
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
    madau_dickinson_time_delayed_redshift_distribution,
)
from astrogwb.populations.registry import ComponentRegistry

__all__ = [
    "RedshiftFn",
    "RedshiftLaw",
    "RedshiftModel",
    "build_redshift_model",
    "known_redshift_models",
    "register_redshift_model",
]


class RedshiftLaw(NamedTuple):
    """One redshift law at one set of hyperparameters.

    ``distribution`` is what the ``redshift`` site is drawn from.
    ``luminosity_distance`` is the *standard* distance on the law's cosmology;
    the population applies any modified propagation on top of it. For a
    physical law ``distribution.total_merger_rate()`` is the observer-frame
    total rate.
    """

    distribution: dist.Distribution
    luminosity_distance: Callable[[ArrayLike], jax.Array]


type RedshiftFn = Callable[[Mapping[str, ArrayLike]], RedshiftLaw]


class RedshiftModel(NamedTuple):
    """A built redshift model: its law, and whether the law has a physical rate."""

    law: RedshiftFn
    has_merger_rate: bool


_REDSHIFT_MODELS = ComponentRegistry("redshift model")
register_redshift_model = _REDSHIFT_MODELS.register
build_redshift_model = _REDSHIFT_MODELS.build


def known_redshift_models() -> tuple[str, ...]:
    """Every registered redshift model name, in sorted order."""
    return _REDSHIFT_MODELS.names()


def _madau_dickinson_distribution(
    params: Mapping[str, ArrayLike],
    *,
    minimum_redshift: float,
    maximum_redshift: float,
    n_grid: int,
) -> RedshiftDistribution:
    return MadauDickinsonRedshiftDistribution(
        params=params,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
        n_grid=n_grid,
    )


@register_redshift_model("madau_dickinson")
def _madau_dickinson(
    *, minimum_redshift: float, maximum_redshift: float, n_grid: int
) -> RedshiftModel:
    r"""The Madau-Dickinson merger-rate density.

    ``params`` must carry ``H0``, ``Omega_m``, ``gamma``, ``kappa`` and
    ``z_peak`` (and ``local_merger_rate`` for the total rate). The window and
    grid are the one the cosmology integrals and the normalization run on.
    """

    def law(params: Mapping[str, ArrayLike]) -> RedshiftLaw:
        distribution = _madau_dickinson_distribution(
            params,
            minimum_redshift=minimum_redshift,
            maximum_redshift=maximum_redshift,
            n_grid=n_grid,
        )
        return RedshiftLaw(distribution, distribution.luminosity_distance)

    return RedshiftModel(law, has_merger_rate=True)


@register_redshift_model("madau_dickinson_uniform_guard")
def _madau_dickinson_uniform_guard(
    *,
    minimum_redshift: float,
    maximum_redshift: float,
    n_grid: int,
    uniform_mixing_fraction: float,
) -> RedshiftModel:
    r"""A Madau-Dickinson law blended with a uniform-in-redshift guard.

    A *proposal* density, not a physical one: mixing a fraction :math:`\epsilon`
    of uniform draws into the Madau-Dickinson density fattens the tails, so
    reweighting to a target far from the generating parameters keeps a usable
    effective sample size. The luminosity distance is the Madau-Dickinson
    component's cosmology -- the mixture changes which redshifts are drawn, not
    how far away a source at a given redshift is. It declares no merger rate.
    """
    if not 0.0 <= uniform_mixing_fraction <= 1.0:
        raise ValueError(
            f"uniform_mixing_fraction must lie in [0, 1], got {uniform_mixing_fraction}"
        )

    def law(params: Mapping[str, ArrayLike]) -> RedshiftLaw:
        base = _madau_dickinson_distribution(
            params,
            minimum_redshift=minimum_redshift,
            maximum_redshift=maximum_redshift,
            n_grid=n_grid,
        )
        mixture = dist.MixtureGeneral(
            dist.Categorical(
                probs=jnp.array(
                    [1.0 - uniform_mixing_fraction, uniform_mixing_fraction]
                )
            ),
            [base, dist.Uniform(minimum_redshift, maximum_redshift)],
            support=base.support,
        )
        return RedshiftLaw(mixture, base.luminosity_distance)

    return RedshiftModel(law, has_merger_rate=False)


@register_redshift_model("madau_dickinson_time_delayed")
def _madau_dickinson_time_delayed(
    *,
    minimum_redshift: float,
    maximum_redshift: float,
    n_grid: int,
    minimum_delay: float,
    maximum_formation_redshift: float,
    n_delay_nodes: int,
) -> RedshiftModel:
    r"""The Madau-Dickinson law read as a formation rate, with delayed mergers.

    A merger follows formation after :math:`\tau \sim p(\tau) \propto
    \tau^{\alpha}`, with :math:`\tau` at least ``minimum_delay`` Gyr and
    :math:`\alpha` ``params["delay_slope"]``. The delay is built inside the law
    so ``delay_slope`` is a traced hyperparameter like ``z_peak``; the floor and
    the quadrature (``n_delay_nodes``) are construction kwargs.

    The delay has no ceiling of its own: its upper bound is the lookback time
    to ``maximum_formation_redshift``, the longest delay any merger can have.
    Each merger redshift then integrates only the delays available to it (see
    :class:`~astrogwb.distributions.redshift.TimeDelayedRedshiftDistribution`),
    so the bound only has to be finite for :math:`\alpha > -1` to normalize. It
    follows ``H0`` and ``Omega_m``, so ``H0`` no longer factors out of the
    spectrum: the delay is in Gyr while lookback time scales as :math:`1/H_0`.
    """

    def law(params: Mapping[str, ArrayLike]) -> RedshiftLaw:
        longest_delay = lookback_time(
            maximum_formation_redshift, params["H0"], params["Omega_m"]
        )
        distribution = madau_dickinson_time_delayed_redshift_distribution(
            params=params,
            time_delay_distribution=PowerLawDelayDistribution(
                params["delay_slope"], minimum_delay, longest_delay
            ),
            n_delay_nodes=n_delay_nodes,
            maximum_formation_redshift=maximum_formation_redshift,
            minimum_redshift=minimum_redshift,
            maximum_redshift=maximum_redshift,
            n_grid=n_grid,
        )
        return RedshiftLaw(distribution, distribution.luminosity_distance)

    return RedshiftModel(law, has_merger_rate=True)
