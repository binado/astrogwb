"""The population stage: one Poisson realization of every source, per key.

A realization owns every stochastic property of its sources: the
hyperparameters (fixed or drawn once), the Poisson count
:math:`N \\sim \\mathrm{Poisson}(\\mathcal{R}\\, T)`, the intrinsic parameters
and inclination the population model draws, and the extrinsic angles drawn
by the joint model -- an isotropic direction (``declination``,
``right_ascension``), the polarization angle ``polarization_angle`` and the
coalescence phase ``coa_phase`` and geocentric ``arrival_time`` in seconds. The downstream stages draw nothing.

One key splits into independent hyperparameter, count and source streams.
Sources are drawn in one batch by
:func:`~astrogwb.simulators.simulate_sources`, so a realization depends on its
count; only waveform evaluation works in fixed-size chunks.
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
from astrogwb.populations.joint import EXTRINSIC_SITES, joint_model
from astrogwb.populations.metadata import AnyPopulation
from astrogwb.simulators.sources import simulate_sources
from astrogwb.utils import require_current_version, years_to_seconds

__all__ = [
    "Hyperparameter",
    "PopulationArtifact",
    "RealizationMetadata",
    "simulate_population",
]

#: One hyperparameter: a fixed value, or the prior it is drawn from once per
#: realization.
type Hyperparameter = float | DistributionConfig


class RealizationMetadata(BaseModel):
    """Everything that determines a population realization, but its draw.

    ``hyperparameters`` holds, per name, either a number -- every realization
    is made at that value -- or a
    :class:`~astrogwb.distributions.config.DistributionConfig` it is drawn
    from, once per realization. ``observation_time`` is
    in years and sets the Poisson mean :math:`\\mathcal{R}\\, T` and, downstream,
    the spectrum normalization.

    The key and requested segment duration are :func:`simulate_population`'s
    inputs, not part of this record; :class:`SimulationConfig` records the
    segment duration for an end-to-end run.
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

    metadata: RealizationMetadata
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


def simulate_population(
    metadata: RealizationMetadata,
    key: jax.Array,
    *,
    segment_duration: float | None = None,
) -> PopulationArtifact:
    """The realization of ``metadata`` at a single typed JAX key.

    Parameters
    ----------
    metadata
        The realization record.
    key
        A single typed JAX key, e.g. from
        :func:`~astrogwb.simulators.realization_key`.
    segment_duration
        Requested segment duration in seconds. Arrival times are sampled
        relative to segment center using the effective duration
        ``T / floor(T / segment_duration)``. ``None`` uses the full observation.
        Supply the same value to the subsequent background stage.

    Raises
    ------
    ValueError
        If ``key`` is not a single key, or ``metadata`` is for another astrogwb
        version, or the requested segment duration is invalid.
    """
    # Set precision before creating arrays or initializing the backend.
    jax.config.update("jax_enable_x64", True)
    require_current_version(metadata.version)
    if key.shape != ():
        raise ValueError(f"key must be a single key, got shape {key.shape}")
    seconds = metadata.observation_seconds
    if segment_duration is not None:
        if not np.isfinite(segment_duration) or not 0 < segment_duration <= seconds:
            raise ValueError(
                "segment_duration must be positive and no longer than the observation"
            )
        seconds /= int(seconds // segment_duration)
    bounds = (-seconds / 2, seconds / 2)
    theta_key, count_key, source_key = jax.random.split(key, 3)
    with handlers.block():
        theta = dict(handlers.seed(metadata.prior_model(), theta_key)())
    redshift, source_model = metadata.population.build()(theta)
    rate = jnp.reshape(redshift.total_merger_rate(), ())
    count = int(jax.random.poisson(count_key, rate * metadata.observation_seconds))
    # One sample preserves column names even for an empty realization.
    sources = simulate_sources(
        joint_model(redshift, source_model, arrival_time_bounds=bounds),
        source_key,
        max(count, 1),
    )
    return PopulationArtifact(
        metadata=metadata,
        rng_key=tuple(int(v) for v in np.asarray(jax.random.key_data(key))),
        total_merger_rate=float(rate),
        hyperparameters={name: float(value) for name, value in theta.items()},
        source_parameters={name: values[:count] for name, values in sources.items()},
    )
