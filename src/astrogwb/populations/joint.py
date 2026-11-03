"""Compose a population's two halves into the model a catalog is drawn from."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.populations._types import PopulationModel, SourceModel

__all__ = [
    "DEFAULT_DENSITY_SITES",
    "EXTRINSIC_SITES",
    "INTRINSIC_DENSITY_SITES",
    "REDSHIFT_SITE",
    "extrinsic_priors",
    "fix_extrinsic_parameters",
    "joint_model",
]

#: The redshift site every population must declare; its density is never
#: excluded.
REDSHIFT_SITE = "redshift"

#: The intrinsic density factors of a source model: the ordered source-frame
#: mass pair. These are the factors an importance weight divides by, since
#: redshift is integrated by quadrature rather than reweighted.
INTRINSIC_DENSITY_SITES: tuple[str, ...] = (
    "source_frame_mass_1",
    "source_frame_mass_2",
)

#: Density factors of the joint model a catalog selects when nothing narrower
#: is requested: ``redshift`` plus the intrinsic ones.
DEFAULT_DENSITY_SITES: tuple[str, ...] = (REDSHIFT_SITE, *INTRINSIC_DENSITY_SITES)

#: The extrinsic sites :func:`joint_model` samples.
EXTRINSIC_SITES = (
    "declination",
    "right_ascension",
    "polarization_angle",
    "coa_phase",
    "arrival_fraction",
)


def extrinsic_priors() -> dict[str, dist.Distribution]:
    """The prior of each :data:`EXTRINSIC_SITES` entry.

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
        "arrival_fraction": dist.Uniform(-0.5, 0.5),
    }


def fix_extrinsic_parameters(
    model: PopulationModel, value: float = 0.0
) -> PopulationModel:
    """``model`` with every extrinsic site fixed at ``value``.

    Fixed sites draw no random numbers, so the intrinsic draws and any Sobol
    coordinates are those of a model without extrinsics.

    Parameters
    ----------
    model
        NumPyro model.
    value
        Value to evaluate.
    """
    return handlers.condition(
        model, data={name: jnp.asarray(value) for name in EXTRINSIC_SITES}
    )


def joint_model(
    redshift_distribution: RedshiftDistribution, source_model: SourceModel
) -> PopulationModel:
    """The model that samples distances, intrinsic sources and extrinsics.

    ``redshift_distribution.distance_model()`` contributes the ``redshift``
    site and the ``luminosity_distance`` deterministic; ``source_model`` the
    intrinsic sites. This model adds isotropic sky coordinates, polarization
    angle, coalescence phase and dimensionless ``arrival_fraction`` uniform
    on ``[-0.5, 0.5)`` as sample sites. The latter locates a source relative
    to its segment center when multiplied by the effective segment duration.
    Identical extrinsic priors cancel in importance ratios. The declared sites
    define all source columns. Call it as ``joint_model(*population(params))``.

    Parameters
    ----------
    redshift_distribution
        The population's redshift law, built at its hyperparameters.
    source_model
        The intrinsic model; it must not declare ``redshift`` or
        ``luminosity_distance``, sky coordinates, polarization angle or coalescence
        phase or arrival fraction.
    """
    distance_model = redshift_distribution.distance_model()

    def model() -> None:
        distance_model()
        source_model()
        for name, prior in extrinsic_priors().items():
            numpyro.sample(name, prior)

    return model
