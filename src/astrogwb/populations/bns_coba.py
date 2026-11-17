r"""BNS population: one NumPyro model for every Madau-Dickinson variant.

The redshift, mass and propagation laws are construction settings of one model,
so the variants share one source density. ``source_model()`` covers only the
source-frame intrinsic parameters.

Unchecked contracts:

- ``parameters`` carries ``H0``, ``Omega_m``, ``gamma``, ``kappa``,
  ``z_peak`` and ``local_merger_rate`` (:math:`\mathrm{Gpc}^{-3}\,
  \mathrm{yr}^{-1}`), plus ``minimum_mass`` and ``mass_width`` for
  ``mass_model="uniform"`` or ``mass_mean`` and ``mass_sigma`` for
  ``mass_model="gaussian"``.
- Modified GW propagation is applied whenever ``xi_0`` is in ``parameters``
  (``xi_n`` must then be too); :math:`\Xi_0 = 1` is plain propagation.
- ``local_merger_rate`` is a pure overall scaling of the spectrum.
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
from typing import Annotated, Literal

import jax
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pydantic import Field

from astrogwb.distributions import UniformCosineDistribution
from astrogwb.populations._types import Parameters, Population, SourceModel
from astrogwb.populations.base import PopulationMetadata
from astrogwb.populations.mass.gaussian_mass_pair import gaussian_mass_pair_model
from astrogwb.populations.mass.uniform_mass_pair import uniform_mass_pair_model
from astrogwb.rates import madau_dickinson_rate

__all__ = ["BNSCobaPopulation"]

_MASS_MODELS: Mapping[str, Callable[[Parameters], tuple[jax.Array, jax.Array]]] = {
    "uniform": uniform_mass_pair_model,
    "gaussian": gaussian_mass_pair_model,
}


class BNSCobaPopulation(PopulationMetadata):
    r"""The BNS population's construction settings.

    See the module docstring for the hyperparameters each setting requires.
    :meth:`build` returns a population exposing ``merger_rate(redshift,
    parameters)`` and ``source_model(parameters)``. Physical simulation builds
    a redshift distribution from the rate and combines its distance model with
    the intrinsic source model and extrinsic priors.

    Attributes
    ----------
    minimum_redshift, maximum_redshift, n_grid
        The grid the cosmology integrals and the redshift normalization run on.
    mass_model
        ``"uniform"`` or ``"gaussian"`` ordered component masses.
    maximum_tidal_deformability, maximum_aligned_spin_component
        Bounds of the uniform tidal-deformability and aligned-spin priors.
    """

    model_name: Literal["bns_coba"] = "bns_coba"
    mass_model: Literal["uniform", "gaussian"]
    n_grid: Annotated[int, Field(gt=1)] = 1000
    maximum_tidal_deformability: Annotated[float, Field(gt=0.0)] = 2000.0
    maximum_aligned_spin_component: Annotated[float, Field(gt=0.0)] = 0.05

    def _build(self) -> Population:
        return _bns_coba_population(self)


class _BNSCobaModel:
    """A shared merger rate and intrinsic model for both pipelines."""

    def __init__(self, config: BNSCobaPopulation) -> None:
        self.config = config

    def merger_rate(self, redshift: ArrayLike, parameters: Parameters) -> jax.Array:
        return madau_dickinson_rate(
            redshift,
            parameters["gamma"],
            parameters["kappa"],
            parameters["z_peak"],
            parameters.get("local_merger_rate", 1.0),
        )

    def source_model(self, parameters: Parameters) -> SourceModel:
        config = self.config
        draw_masses = _MASS_MODELS[config.mass_model]

        def model() -> None:
            draw_masses(parameters)
            spin_dist = dist.Uniform(
                -config.maximum_aligned_spin_component,
                config.maximum_aligned_spin_component,
                validate_args=True,
            )
            numpyro.sample("spin_1z", spin_dist)
            numpyro.sample("spin_2z", spin_dist)
            lambda_dist = dist.Uniform(
                0.0, config.maximum_tidal_deformability, validate_args=True
            )
            numpyro.sample("lambda_1", lambda_dist)
            numpyro.sample("lambda_2", lambda_dist)
            numpyro.sample("inclination", UniformCosineDistribution(validate_args=True))

        return model


def _bns_coba_population(config: BNSCobaPopulation) -> Population:
    return _BNSCobaModel(config)
