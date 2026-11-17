"""Population metadata and prior helpers for scientific configurations."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Annotated

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from pydantic import BaseModel, ConfigDict, Field

from astrogwb import __version__
from astrogwb.distributions.config import DistributionConfig
from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.populations.metadata import AnyPopulation
from astrogwb.utils import years_to_seconds

__all__ = [
    "EXTRINSIC_SITES",
    "Hyperparameter",
    "RealizationMetadata",
    "extrinsic_priors",
    "segmented_seconds",
]

#: One hyperparameter: a fixed value, or the prior it is drawn from once per
#: realization.
type Hyperparameter = float | DistributionConfig

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


class RealizationMetadata(BaseModel):
    """Everything that determines a population realization, but its draw.

    ``hyperparameters`` holds, per name, either a number -- every realization
    is made at that value -- or a
    :class:`~astrogwb.distributions.config.DistributionConfig` it is drawn
    from, once per realization. ``observation_time`` is
    in years and sets the Poisson mean :math:`\\mathcal{R}\\, T` and, downstream,
    the spectrum normalization.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    population: AnyPopulation
    hyperparameters: dict[str, Hyperparameter]
    observation_time: Annotated[float, Field(gt=0.0)]
    version: str = __version__

    @property
    def observation_seconds(self) -> float:
        """The observation time in seconds."""
        return years_to_seconds(self.observation_time)

    @property
    def fixed(self) -> dict[str, float]:
        """The hyperparameters every realization shares, by name."""
        return {
            name: float(value)
            for name, value in self.hyperparameters.items()
            if not isinstance(value, DistributionConfig)
        }

    @property
    def sampled(self) -> dict[str, DistributionConfig]:
        """The hyperparameters drawn once per realization, with their priors."""
        return {
            name: value
            for name, value in self.hyperparameters.items()
            if isinstance(value, DistributionConfig)
        }

    def prior_model(self) -> Callable[[], dict[str, jax.Array]]:
        """A zero-argument NumPyro model of the hyperparameters of one realization.

        Sampled hyperparameters are ``numpyro.sample`` sites; fixed ones are
        ``numpyro.deterministic`` sites. The model returns every hyperparameter
        by name, so it composes with ``Predictive``, ``handlers.condition`` and
        ``log_density``.
        """

        declared = {
            name: self.hyperparameters[name] for name in sorted(self.hyperparameters)
        }
        priors = {
            name: value.build()
            for name, value in declared.items()
            if isinstance(value, DistributionConfig)
        }

        def model() -> dict[str, jax.Array]:
            theta: dict[str, jax.Array] = {}
            for name, value in declared.items():
                if name in priors:
                    theta[name] = jnp.asarray(numpyro.sample(name, priors[name]))
                else:
                    theta[name] = jnp.asarray(
                        numpyro.deterministic(
                            name, jnp.asarray(value, dtype=jnp.float64)
                        )
                    )
            return theta

        return model


def segmented_seconds(observation_seconds: float, segment_duration: float) -> float:
    """The time the whole segments of ``segment_duration`` cover, in seconds.

    ``floor(observation_seconds / segment_duration)`` segments of exactly
    ``segment_duration``: the remainder shorter than a segment is not observed.

    Parameters
    ----------
    observation_seconds
        The observation time in seconds.
    segment_duration
        The segment length in seconds.
    """
    return segment_duration * int(observation_seconds // segment_duration)
