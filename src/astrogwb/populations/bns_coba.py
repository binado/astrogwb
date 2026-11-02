r"""BNS population: one NumPyro model for every Madau-Dickinson variant.

The redshift, mass and propagation laws are construction settings of one model,
so the variants share one source density. ``source_model()`` covers only the
source-frame intrinsic parameters.

Unchecked contracts:

- ``parameters`` carries ``H0``, ``Omega_m``, ``gamma``, ``kappa``,
  ``z_peak`` and ``local_merger_rate`` (:math:`\mathrm{Gpc}^{-3}\,
  \mathrm{yr}^{-1}`), plus ``minimum_mass`` and ``mass_width`` for
  ``mass_model="uniform"`` or ``mass_mean`` and ``mass_sigma`` for
  ``mass_model="gaussian"``, plus ``delay_slope`` when ``time_delay=True``.
- Modified GW propagation is applied whenever ``xi_0`` is in ``parameters``
  (``xi_n`` must then be too); :math:`\Xi_0 = 1` is plain propagation.
- ``local_merger_rate`` is a pure overall scaling of the spectrum; ``H0`` is
  too, except when ``time_delay=True``: the delay is in Gyr while lookback time
  scales as :math:`1/H_0`, so the normalized redshift law depends on ``H0``.
- Component masses are an ordered pair: the first is the larger one. The
  uniform law has compact support, so a hyperparameter step that moves its
  edges can send catalog samples outside it; the Gaussian law has none.
- Inclination is always sampled, isotropically, and the intrinsic sites (masses,
  spins, tidal deformabilities) are independent of inclination; independence
  of redshift is structural, since the source model never sees it.
  :mod:`astrogwb.importance.spectrum` relies on both.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Literal, Self

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from pydantic import Field, model_validator

from astrogwb.cosmology import lookback_time
from astrogwb.distributions import UniformCosineDistribution
from astrogwb.distributions.delay import PowerLawDelayDistribution
from astrogwb.distributions.redshift import MadauDickinsonRedshiftDistribution
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.distributions.redshift.madau_dickinson import (
    madau_dickinson_time_delayed_redshift_distribution,
)
from astrogwb.populations._types import Parameters, Population, SourceModel
from astrogwb.populations.base import PopulationMetadata
from astrogwb.populations.mass.gaussian_mass_pair import gaussian_mass_pair_model
from astrogwb.populations.mass.uniform_mass_pair import uniform_mass_pair_model

__all__ = ["BNSCobaPopulation"]

_MASS_MODELS: Mapping[str, Callable[[Parameters], tuple[jax.Array, jax.Array]]] = {
    "uniform": uniform_mass_pair_model,
    "gaussian": gaussian_mass_pair_model,
}


class BNSCobaPopulation(PopulationMetadata):
    r"""The BNS population's construction settings.

    See the module docstring for the hyperparameters each setting requires.
    :meth:`build` returns ``parameters -> (redshift_distribution,
    source_model)``. The distribution's ``total_merger_rate()`` is the
    observer-frame rate in mergers per second, shape ``()``, and its
    ``distance_model()`` samples redshift and GW distance; ``source_model()``
    declares the intrinsic sites only.
    :func:`~astrogwb.populations.joint.joint_model` composes the two.

    Attributes
    ----------
    minimum_redshift, maximum_redshift, n_grid
        The grid the cosmology integrals and the redshift normalization run on.
    mass_model
        ``"uniform"`` or ``"gaussian"`` ordered component masses.
    time_delay
        Read the Madau-Dickinson law as the *formation* rate and delay mergers
        by :math:`p(\tau) \propto \tau^{\alpha}`; requires ``minimum_delay``
        (Gyr), and ``delay_slope`` in the hyperparameters.
    minimum_delay, maximum_formation_redshift, n_delay_nodes
        The delay floor, the formation cut-off and the order of the delay
        quadrature. Only read when ``time_delay`` is true.
    maximum_tidal_deformability, maximum_aligned_spin_component
        Bounds of the uniform tidal-deformability and aligned-spin priors.
    """

    model_name: Literal["bns_coba"] = "bns_coba"
    mass_model: Literal["uniform", "gaussian"]
    time_delay: bool = False
    minimum_delay: Annotated[float, Field(gt=0.0)] | None = None
    maximum_formation_redshift: Annotated[float, Field(gt=0.0)] = 20.0
    n_delay_nodes: Annotated[int, Field(gt=0)] = 48
    n_grid: Annotated[int, Field(gt=1)] = 1000
    maximum_tidal_deformability: Annotated[float, Field(gt=0.0)] = 2000.0
    maximum_aligned_spin_component: Annotated[float, Field(gt=0.0)] = 0.05

    @model_validator(mode="after")
    def _delay_needs_a_floor(self) -> Self:
        if self.time_delay and self.minimum_delay is None:
            raise ValueError("time_delay=True requires minimum_delay")
        return self

    def _build(self) -> Population:
        return _bns_coba_population(self)


def _redshift_distribution(
    parameters: Parameters, config: BNSCobaPopulation
) -> RedshiftDistribution:
    """The Madau-Dickinson law of ``config``, delayed when ``config.time_delay``.

    ``delay_slope`` is traced; the floor and quadrature are construction
    settings. The delay's upper bound is the lookback time to
    ``maximum_formation_redshift``, so it follows ``H0`` and ``Omega_m``.
    """
    if not config.time_delay:
        return MadauDickinsonRedshiftDistribution(
            params=parameters,
            minimum_redshift=config.minimum_redshift,
            maximum_redshift=config.maximum_redshift,
            n_grid=config.n_grid,
        )
    assert config.minimum_delay is not None
    longest_delay = lookback_time(
        config.maximum_formation_redshift, parameters["H0"], parameters["Omega_m"]
    )
    return madau_dickinson_time_delayed_redshift_distribution(
        params=parameters,
        time_delay_distribution=PowerLawDelayDistribution(
            parameters["delay_slope"], config.minimum_delay, longest_delay
        ),
        n_delay_nodes=config.n_delay_nodes,
        maximum_formation_redshift=config.maximum_formation_redshift,
        minimum_redshift=config.minimum_redshift,
        maximum_redshift=config.maximum_redshift,
        n_grid=config.n_grid,
    )


def _bns_coba_population(config: BNSCobaPopulation) -> Population:
    """The population ``config`` describes."""
    draw_masses = _MASS_MODELS[config.mass_model]
    spin_bound = config.maximum_aligned_spin_component

    def _redshift_distribution_and_source_model(
        parameters: Parameters,
    ) -> tuple[RedshiftDistribution, SourceModel]:
        def _source_model() -> None:
            mass_1, _ = draw_masses(parameters)

            spin_dist = dist.Uniform(-spin_bound, spin_bound, validate_args=True)
            numpyro.sample("spin_1z", spin_dist)
            numpyro.sample("spin_2z", spin_dist)

            lambda_dist = dist.Uniform(
                0.0, config.maximum_tidal_deformability, validate_args=True
            )
            numpyro.sample("lambda_1", lambda_dist)
            numpyro.sample("lambda_2", lambda_dist)

            # Time-aligned; inclination is a separate stochastic site.
            zeros = jnp.zeros_like(mass_1)
            numpyro.deterministic("coa_time", zeros)
            numpyro.sample("inclination", UniformCosineDistribution(validate_args=True))

        return _redshift_distribution(parameters, config), _source_model

    return _redshift_distribution_and_source_model
