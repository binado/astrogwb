"""Compose a population's two halves into the model a catalog is drawn from."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.populations._types import PopulationModel, SourceModel

__all__ = ["joint_model"]


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
        declination = dist.TransformedDistribution(
            UniformCosineDistribution(),
            dist.transforms.AffineTransform(-jnp.pi / 2.0, 1.0),
        )
        outputs = {
            **sources,
            "declination": numpyro.sample("declination", declination),
            "right_ascension": numpyro.sample(
                "right_ascension", dist.Uniform(0.0, 2.0 * jnp.pi)
            ),
            "polarization_angle": numpyro.sample(
                "polarization_angle", dist.Uniform(0.0, jnp.pi)
            ),
            "coa_phase": numpyro.sample("coa_phase", dist.Uniform(0.0, 2.0 * jnp.pi)),
        }
        return {name: jnp.asarray(value) for name, value in outputs.items()}

    return model
