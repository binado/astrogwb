"""Shared names and priors for source parameters."""

from __future__ import annotations

import math

import numpyro.distributions as dist

from astrogwb.distributions.orientation import UniformCosineDistribution

__all__ = [
    "DEFAULT_DENSITY_SITES",
    "EXTRINSIC_SITES",
    "INTRINSIC_DENSITY_SITES",
    "REDSHIFT_SITE",
    "extrinsic_priors",
]

#: The redshift site, whose density is never excluded.
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

#: The extrinsic sites sampled by physical simulations.
EXTRINSIC_SITES = (
    "declination",
    "right_ascension",
    "polarization_angle",
    "coa_phase",
    "coa_time",
)


def extrinsic_priors(
    coa_time_bounds: tuple[float, float] = (-0.5, 0.5),
) -> dict[str, dist.Distribution]:
    """The prior of each :data:`EXTRINSIC_SITES` entry.

    Declination is the cosine law shifted from colatitude to
    :math:`[-\\pi/2, \\pi/2]`. Built on call: constructing distributions
    creates arrays, which must not happen at import.

    Parameters
    ----------
    coa_time_bounds
        Lower and upper geocentric arrival times in seconds.
    """
    return {
        "declination": dist.TransformedDistribution(
            UniformCosineDistribution(),
            dist.transforms.AffineTransform(-math.pi / 2.0, 1.0),
        ),
        "right_ascension": dist.Uniform(0.0, 2.0 * math.pi),
        "polarization_angle": dist.Uniform(0.0, math.pi),
        "coa_phase": dist.Uniform(0.0, 2.0 * math.pi),
        "coa_time": dist.Uniform(*coa_time_bounds),
    }
