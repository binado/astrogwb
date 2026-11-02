"""The population stage: one Poisson realization of every source, per key.

A realization owns every stochastic property of its sources: the
hyperparameters (fixed or drawn once), the Poisson count
:math:`N \\sim \\mathrm{Poisson}(\\mathcal{R}\\, T)`, the intrinsic parameters
and inclination the population model draws, and the extrinsic angles drawn
by the joint model -- an isotropic direction (``declination``,
``right_ascension``), the polarization angle ``polarization_angle`` and the
coalescence phase ``coa_phase``. The downstream stages draw nothing.

One key splits into independent hyperparameter, count and source streams.
Sources are drawn in one batch, so a realization depends on its count; only
waveform evaluation works in fixed-size chunks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpy.typing import NDArray
from numpyro import handlers
from pydantic import BaseModel, ConfigDict, Field

from astrogwb import __version__
from astrogwb.distributions.config import DistributionConfig
from astrogwb.populations import joint_model
from astrogwb.populations.joint import EXTRINSIC_SITES
from astrogwb.populations.metadata import PopulationMetadata as PopulationSpec
from astrogwb.utils import years_to_seconds
from astrogwb.utils.numpyro import sample_and_logprob

__all__ = [
    "Hyperparameter",
    "PopulationArtifact",
    "PopulationMetadata",
    "PopulationSimulator",
]

#: One hyperparameter: a fixed value, or the prior it is drawn from once per
#: realization.
type Hyperparameter = float | DistributionConfig


class PopulationMetadata(BaseModel):
    """Everything that determines a population realization, but its draw.

    ``hyperparameters`` holds, per name, either a number -- every realization
    is made at that value -- or a
    :class:`~astrogwb.distributions.config.DistributionConfig` it is drawn
    from, once per realization. ``observation_time`` is
    in years and sets the Poisson mean :math:`\\mathcal{R}\\, T` and, downstream,
    the spectrum normalization.

    The key is the simulator's input, not part of the record.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    population: PopulationSpec
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


@dataclass(frozen=True)
class PopulationArtifact:
    """One population realization, in memory.

    ``source_parameters`` holds one ``(count,)`` column per source parameter,
    the extrinsic columns :data:`~astrogwb.populations.joint.EXTRINSIC_SITES` included; source ``i`` has
    the stable ID ``i``. ``total_merger_rate`` is the realized
    :math:`\\mathcal{R}` in events per second and ``hyperparameters`` the
    realized values. ``rng_key`` is the key's data, the realization's
    provenance.
    """

    metadata: PopulationMetadata
    rng_key: tuple[int, ...]
    total_merger_rate: float
    hyperparameters: dict[str, float]
    source_parameters: dict[str, NDArray[Any]]

    def __post_init__(self) -> None:
        """Require the extrinsic columns."""
        missing = set(EXTRINSIC_SITES) - self.source_parameters.keys()
        if missing:
            raise ValueError(
                f"population missing coordinate columns {sorted(missing)}; regenerate with astrogwb simulate"
            )

    @property
    def count(self) -> int:
        """The realized number of sources."""
        return int(next(iter(self.source_parameters.values())).shape[0])

    @property
    def observation_seconds(self) -> float:
        """The observation time in seconds."""
        return self.metadata.observation_seconds


class PopulationSimulator:
    """Draw a full population realization for each input key."""

    def __init__(self, metadata: PopulationMetadata) -> None:

        # Set precision before creating arrays or initializing the backend.
        jax.config.update("jax_enable_x64", True)
        if metadata.version != __version__:
            raise ValueError(
                f"metadata is for astrogwb {metadata.version}, but {__version__} "
                "is installed"
            )
        self._metadata = metadata
        self._population = metadata.population.build()
        self._prior_model = metadata.prior_model()

    @property
    def metadata(self) -> PopulationMetadata:
        """The record this simulator draws."""
        return self._metadata

    def __call__(self, key: jax.Array) -> PopulationArtifact:
        """Draw the realization at a single typed JAX key."""

        if key.shape != ():
            raise ValueError(f"key must be a single key, got shape {key.shape}")
        theta_key, count_key, source_key = jax.random.split(key, 3)
        with handlers.block():
            theta = dict(handlers.seed(self._prior_model, theta_key)())
        redshift, source_model = self._population(theta)
        rate = jnp.reshape(redshift.total_merger_rate(), ())
        count = int(
            jax.random.poisson(count_key, rate * self.metadata.observation_seconds)
        )
        # One sample preserves column names even for an empty realization.
        sources, _ = sample_and_logprob(
            joint_model(redshift, source_model),
            source_key,
            num_samples=max(count, 1),
            density_sites=(),
        )
        return PopulationArtifact(
            metadata=self.metadata,
            rng_key=tuple(int(v) for v in np.asarray(jax.random.key_data(key))),
            total_merger_rate=float(rate),
            hyperparameters={name: float(value) for name, value in theta.items()},
            source_parameters={
                name: np.asarray(values)[:count] for name, values in sources.items()
            },
        )
