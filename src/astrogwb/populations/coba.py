r"""BNS intrinsic source model of the COBA population.

``source_model`` covers only the source-frame intrinsic parameters, and the
caller supplies redshift, distance and the merger rate. For the rate, use
:func:`astrogwb.rates.madau_dickinson_rate` directly.

Unchecked contracts:

- ``parameters`` carries ``minimum_mass`` and ``mass_width`` for
  ``mass_model="uniform"``, or ``mass_mean`` and ``mass_sigma`` for
  ``mass_model="gaussian"``.
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
from typing import Literal

import jax
import numpyro
import numpyro.distributions as dist

from astrogwb.distributions import UniformCosineDistribution
from astrogwb.populations._types import Parameters, SourceModel
from astrogwb.populations.mass.gaussian_mass_pair import gaussian_mass_pair_model
from astrogwb.populations.mass.uniform_mass_pair import uniform_mass_pair_model

__all__ = ["source_model"]

_MASS_MODELS: Mapping[str, Callable[[Parameters], tuple[jax.Array, jax.Array]]] = {
    "uniform": uniform_mass_pair_model,
    "gaussian": gaussian_mass_pair_model,
}


def source_model(
    parameters: Parameters,
    *,
    mass_model: Literal["uniform", "gaussian"],
    maximum_tidal_deformability: float = 2000.0,
    maximum_aligned_spin_component: float = 0.05,
) -> SourceModel:
    """Component masses, aligned spins, tidal deformabilities and inclination.

    Parameters
    ----------
    parameters
        Hyperparameter values; see the module docstring.
    mass_model
        ``"uniform"`` or ``"gaussian"`` ordered component masses.
    maximum_tidal_deformability
        Upper bound of the uniform tidal-deformability prior.
    maximum_aligned_spin_component
        Bound of the uniform aligned-spin prior.

    Returns
    -------
    SourceModel
        A no-argument NumPyro model declaring intrinsic sites and inclination.
    """
    draw_masses = _MASS_MODELS[mass_model]

    def model() -> None:
        draw_masses(parameters)
        spin_dist = dist.Uniform(
            -maximum_aligned_spin_component,
            maximum_aligned_spin_component,
            validate_args=True,
        )
        numpyro.sample("spin_1z", spin_dist)
        numpyro.sample("spin_2z", spin_dist)
        lambda_dist = dist.Uniform(0.0, maximum_tidal_deformability, validate_args=True)
        numpyro.sample("lambda_1", lambda_dist)
        numpyro.sample("lambda_2", lambda_dist)
        numpyro.sample("inclination", UniformCosineDistribution(validate_args=True))

    return model
