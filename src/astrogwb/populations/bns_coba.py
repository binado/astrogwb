r"""BNS population: one NumPyro model for every Madau-Dickinson variant.

The redshift law, the mass law, the propagation law and the guard mixture are
construction settings of one model rather than separate registered functions,
so the variants cannot disagree about the source density they share. A
population is a callable ``parameters -> (merger_rate, model)``: the redshift
distribution is built once per call, so the rate and the source density come
from the same grid, and ``model()`` is a no-argument NumPyro model whose sample
sites are exactly the columns a catalog stores.

Contracts the caller is trusted to honour (nothing here checks them):

- ``parameters`` carries ``H0``, ``Omega_m``, ``gamma``, ``kappa``,
  ``z_peak`` and ``local_merger_rate`` (:math:`\mathrm{Gpc}^{-3}\,
  \mathrm{yr}^{-1}`), plus ``minimum_mass`` and ``mass_width`` for
  ``mass_model="uniform"`` or ``mass_mean`` and ``mass_sigma`` for
  ``mass_model="gaussian"``, plus ``delay_slope`` when ``time_delay=True``.
- Modified GW propagation is applied whenever ``xi_0`` is in ``parameters``
  (``xi_n`` must then be too). At :math:`\Xi_0 = 1` the distance ratio is
  identically one, so a configuration that always carries the fiducial
  ``xi_0 = 1`` reproduces plain cosmological propagation exactly.
- ``local_merger_rate`` is a pure overall scaling of the spectrum; ``H0`` is
  too, except when ``time_delay=True``: the delay is in Gyr while lookback time
  scales as :math:`1/H_0`, so the normalized redshift law depends on ``H0``.
- ``uniform_mixing_fraction > 0`` makes a *proposal* density. The returned rate
  is still the Madau-Dickinson total rate, which normalizes the
  Madau-Dickinson density and not a mixture of it with a uniform component, so
  such a population must not be used as an analysis target or an injection.
- Component masses are an ordered pair: the first is the larger one. The
  uniform law has compact support, so a hyperparameter step that moves its
  edges can send catalog samples outside it; the Gaussian law has none.
- Inclination is always sampled, isotropically.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.cosmology import log_gw_em_ratio, lookback_time
from astrogwb.distributions import UniformCosineDistribution
from astrogwb.distributions.delay import PowerLawDelayDistribution
from astrogwb.distributions.redshift import MadauDickinsonRedshiftDistribution
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.distributions.redshift.madau_dickinson import (
    madau_dickinson_time_delayed_redshift_distribution,
)
from astrogwb.populations._types import Parameters, Population, PopulationModel
from astrogwb.populations.mass.gaussian_mass_pair import gaussian_mass_pair_model
from astrogwb.populations.mass.uniform_mass_pair import uniform_mass_pair_model

__all__ = ["bns_coba_population_fn"]

_MASS_MODELS: Mapping[str, Callable[[Parameters], tuple[jax.Array, jax.Array]]] = {
    "uniform": uniform_mass_pair_model,
    "gaussian": gaussian_mass_pair_model,
}


def _time_delayed_redshift_distribution(
    parameters: Parameters,
    *,
    minimum_redshift: float,
    maximum_redshift: float,
    n_grid: int,
    minimum_delay: float,
    maximum_formation_redshift: float,
    n_delay_nodes: int,
) -> RedshiftDistribution:
    """The delayed Madau-Dickinson law, its delay slope read from ``parameters``.

    The delay is built inside the model so ``delay_slope`` is a traced
    hyperparameter like ``z_peak``; the floor and the quadrature are
    construction settings.

    The delay has no ceiling of its own: its upper bound is the lookback time
    to ``maximum_formation_redshift``, the longest delay any merger can have. It
    follows ``H0`` and ``Omega_m``.
    """
    longest_delay = lookback_time(
        maximum_formation_redshift, parameters["H0"], parameters["Omega_m"]
    )
    return madau_dickinson_time_delayed_redshift_distribution(
        params=parameters,
        time_delay_distribution=PowerLawDelayDistribution(
            parameters["delay_slope"], minimum_delay, longest_delay
        ),
        n_delay_nodes=n_delay_nodes,
        maximum_formation_redshift=maximum_formation_redshift,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
        n_grid=n_grid,
    )


def _redshift_distribution(
    parameters: Parameters,
    *,
    minimum_redshift: float,
    maximum_redshift: float,
    n_grid: int,
    time_delay: bool,
    minimum_delay: float | None,
    maximum_formation_redshift: float,
    n_delay_nodes: int,
) -> RedshiftDistribution:
    if not time_delay:
        return MadauDickinsonRedshiftDistribution(
            params=parameters,
            minimum_redshift=minimum_redshift,
            maximum_redshift=maximum_redshift,
            n_grid=n_grid,
        )
    if minimum_delay is None:
        raise ValueError("time_delay=True requires minimum_delay")
    return _time_delayed_redshift_distribution(
        parameters,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
        n_grid=n_grid,
        minimum_delay=minimum_delay,
        maximum_formation_redshift=maximum_formation_redshift,
        n_delay_nodes=n_delay_nodes,
    )


def _redshift_site(
    redshift_distribution: RedshiftDistribution,
    minimum_redshift: float,
    maximum_redshift: float,
    uniform_mixing_fraction: float,
) -> dist.Distribution:
    """The density the ``redshift`` site draws from, optionally with a uniform guard."""
    if uniform_mixing_fraction == 0.0:
        return redshift_distribution
    # Static probabilities, so NumPyro can check the simplex constraint
    # (``0 <= uniform_mixing_fraction <= 1``) eagerly instead of under a trace.
    mixing = dist.Categorical(
        probs=np.array([1.0 - uniform_mixing_fraction, uniform_mixing_fraction]),
        validate_args=True,
    )
    return dist.MixtureGeneral(
        mixing,
        [
            redshift_distribution,
            dist.Uniform(minimum_redshift, maximum_redshift, validate_args=True),
        ],
        support=redshift_distribution.support,
        validate_args=True,
    )


def bns_coba_population_fn(
    minimum_redshift: float,
    maximum_redshift: float,
    mass_model: Literal["uniform", "gaussian"],
    time_delay: bool = False,
    minimum_delay: float | None = None,
    maximum_formation_redshift: float = 20.0,
    n_delay_nodes: int = 48,
    uniform_mixing_fraction: float = 0.0,
    n_grid: int = 1000,
    maximum_tidal_deformability: float = 2000.0,
    maximum_aligned_spin_component: float = 0.05,
) -> Population:
    r"""Build a BNS population from its construction settings.

    The module docstring lists the hyperparameters each setting requires and
    the contracts the caller is trusted to honour.

    Parameters
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
    uniform_mixing_fraction
        Fraction of uniform-in-redshift draws blended into the redshift law, a
        guard that fattens the tails of a proposal. Must lie in ``[0, 1]``.
    maximum_tidal_deformability, maximum_aligned_spin_component
        Bounds of the uniform tidal-deformability and aligned-spin priors.

    Returns
    -------
    Population
        ``parameters -> (merger_rate, model)``. The rate is the observer-frame
        total merger rate in mergers per second, shape ``()``; ``model()``
        declares the source sites and returns the columns a catalog stores.

    Raises
    ------
    ValueError
        If ``mass_model`` is unknown.
    """
    if mass_model not in _MASS_MODELS:
        raise ValueError(
            f"mass_model must be one of {sorted(_MASS_MODELS)}, got {mass_model!r}"
        )
    draw_masses = _MASS_MODELS[mass_model]

    def _merger_rate_and_population_model(
        parameters: Parameters,
    ) -> tuple[jax.Array, PopulationModel]:
        redshift_distribution = _redshift_distribution(
            parameters,
            minimum_redshift=minimum_redshift,
            maximum_redshift=maximum_redshift,
            n_grid=n_grid,
            time_delay=time_delay,
            minimum_delay=minimum_delay,
            maximum_formation_redshift=maximum_formation_redshift,
            n_delay_nodes=n_delay_nodes,
        )
        redshift_site = _redshift_site(
            redshift_distribution,
            minimum_redshift,
            maximum_redshift,
            uniform_mixing_fraction,
        )

        def _population_model() -> dict[str, jax.Array]:
            redshift = jnp.asarray(numpyro.sample("redshift", redshift_site))
            distance = redshift_distribution.luminosity_distance(redshift)
            if "xi_0" in parameters:
                distance = distance * jnp.exp(
                    log_gw_em_ratio(redshift, parameters["xi_0"], parameters["xi_n"])
                )
            luminosity_distance = numpyro.deterministic("luminosity_distance", distance)

            mass_1, mass_2 = draw_masses(parameters)

            spin_dist = dist.Uniform(
                -maximum_aligned_spin_component,
                maximum_aligned_spin_component,
                validate_args=True,
            )
            spin_1z = numpyro.sample("spin_1z", spin_dist)
            spin_2z = numpyro.sample("spin_2z", spin_dist)

            lambda_dist = dist.Uniform(
                0.0, maximum_tidal_deformability, validate_args=True
            )
            lambda_1 = numpyro.sample("lambda_1", lambda_dist)
            lambda_2 = numpyro.sample("lambda_2", lambda_dist)

            one_plus_z = 1.0 + redshift
            detector_frame_mass_1 = numpyro.deterministic(
                "detector_frame_mass_1", mass_1 * one_plus_z
            )
            detector_frame_mass_2 = numpyro.deterministic(
                "detector_frame_mass_2", mass_2 * one_plus_z
            )

            # Phase- and time-aligned; inclination is a separate stochastic site.
            zeros = jnp.zeros_like(redshift)
            coa_phase = numpyro.deterministic("coa_phase", zeros)
            coa_time = numpyro.deterministic("coa_time", zeros)
            inclination = numpyro.sample(
                "inclination", UniformCosineDistribution(validate_args=True)
            )

            sources: dict[str, ArrayLike] = {
                "redshift": redshift,
                "source_frame_mass_1": mass_1,
                "source_frame_mass_2": mass_2,
                "spin_1z": spin_1z,
                "spin_2z": spin_2z,
                "lambda_1": lambda_1,
                "lambda_2": lambda_2,
                "detector_frame_mass_1": detector_frame_mass_1,
                "detector_frame_mass_2": detector_frame_mass_2,
                "luminosity_distance": luminosity_distance,
                "coa_phase": coa_phase,
                "coa_time": coa_time,
                "inclination": inclination,
            }
            return {name: jnp.asarray(values) for name, values in sources.items()}

        return redshift_distribution.total_merger_rate(), _population_model

    return _merger_rate_and_population_model
