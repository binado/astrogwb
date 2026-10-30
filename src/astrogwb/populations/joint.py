"""Compose a population's two halves into the model a catalog is drawn from."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.populations._types import PopulationModel, SourceModel

__all__ = [
    "EXTRINSIC_SITES",
    "extrinsic_priors",
    "fix_extrinsic_parameters",
    "joint_model",
]

#: The extrinsic sites :func:`joint_model` samples.
EXTRINSIC_SITES = (
    "declination",
    "right_ascension",
    "polarization_angle",
    "coa_phase",
)


def extrinsic_priors() -> dict[str, dist.Distribution]:
    """The isotropic prior of each :data:`EXTRINSIC_SITES` entry.

    Declination is the cosine law shifted from colatitude to
    :math:`[-\\pi/2, \\pi/2]`. Built on call: constructing distributions
    creates arrays, which must not happen at import.
    """
    return {
        "declination": dist.TransformedDistribution(
            UniformCosineDistribution(),
            dist.transforms.AffineTransform(-math.pi / 2.0, 1.0),
        ),
        "right_ascension": dist.Uniform(0.0, 2.0 * math.pi),
        "polarization_angle": dist.Uniform(0.0, math.pi),
        "coa_phase": dist.Uniform(0.0, 2.0 * math.pi),
    }


def fix_extrinsic_parameters(
    model: PopulationModel, value: float = 0.0
) -> PopulationModel:
    """``model`` with every extrinsic site fixed at ``value``.

    Fixed sites draw no random numbers, so the intrinsic draws and any Sobol
    coordinates are those of a model without extrinsics.
    """
    return handlers.condition(
        model, data={name: jnp.asarray(value) for name in EXTRINSIC_SITES}
    )


def joint_model(
    redshift_distribution: RedshiftDistribution, source_model: SourceModel
) -> PopulationModel:
    """The model that samples distances, intrinsic sources and extrinsic angles.

    ``redshift_distribution.distance_model()`` contributes the ``redshift``
    site and the ``luminosity_distance`` deterministic; ``source_model`` the
    intrinsic sites. This model adds isotropic sky coordinates, polarization
    angle and coalescence phase as sample sites. The returned mapping holds all
    source columns. Call it as ``joint_model(*population(params))``.

    Parameters
    ----------
    redshift_distribution
        The population's redshift law, built at its hyperparameters.
    source_model
        The intrinsic model; it must not declare ``redshift`` or
        ``luminosity_distance``, sky coordinates, polarization angle or coalescence
        phase.
    """
    distance_model = redshift_distribution.distance_model()

    def model() -> dict[str, jax.Array]:
        sources = {**distance_model(), **source_model()}
        outputs = {
            **sources,
            **{
                name: numpyro.sample(name, prior)
                for name, prior in extrinsic_priors().items()
            },
        }
        return {name: jnp.asarray(value) for name, value in outputs.items()}

    return model
