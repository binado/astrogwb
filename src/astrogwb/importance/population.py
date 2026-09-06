r"""Population models and their evaluated importance-sampling quantities.

A population describes the source density at one hyperparameter point. It
also supplies the luminosity distance governing signal amplitude and the
observer-frame total merger rate. Concrete subclasses own the physics of
those two quantities; the generic population has no propagation parameters.

The target density and distance are evaluated at fixed catalog samples. The
proposal density and reference distance are cached by ``ImportanceCatalog``.
The reference distance must correspond to the stored polarization power,
which scales as distance to the power minus two. It must not be reconstructed
from a cosmology table: interpolation differences would bias every weight.

Only source densities that differ between target and proposal need to be
included, except redshift, which is always required. Both densities must use
the same source-parameter factors. Evaluate only where the proposal has
support: subtracting two negative-infinite log densities produces ``nan``.

Population instances contain distributions and hyperparameters, never catalog
arrays. Concrete dataclass subclasses must each be registered as JAX pytrees
so their distribution and parameter values remain dynamic under transforms.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.distributions.redshift import RedshiftDistribution


@dataclass(frozen=True)
class Population(ABC):
    r"""Source model :math:`p(x \mid \theta)` with distance and rate behavior.

    ``distributions`` must contain a ``RedshiftDistribution`` under
    ``redshift``. Other source-parameter laws may be supplied by name.
    ``params`` carries the hyperparameters needed by the concrete population;
    the base class requires no particular parameter names. Treat both mappings
    as immutable after construction.
    """

    distributions: Mapping[str, dist.Distribution]
    params: Mapping[str, ArrayLike]

    def __post_init__(self) -> None:
        if "redshift" not in self.distributions:
            raise ValueError("Population requires a redshift distribution")
        if not isinstance(self.distributions["redshift"], RedshiftDistribution):
            raise TypeError("Population redshift must be a RedshiftDistribution")

    @property
    def redshift_distribution(self) -> RedshiftDistribution:
        """The redshift law, validated when the population is constructed."""
        return cast(RedshiftDistribution, self.distributions["redshift"])

    def log_prob(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Sum source log densities; missing source parameters raise KeyError."""
        log_probs = {
            name: distribution.log_prob(source_parameters[name])
            for name, distribution in self.distributions.items()
        }
        return jax.tree.reduce(operator.add, log_probs, initializer=jnp.zeros(()))

    @abstractmethod
    def luminosity_distance(self, redshift: ArrayLike) -> jax.Array:
        """Distance governing waveform amplitude, in Mpc, at each redshift."""

    @abstractmethod
    def total_merger_rate(self) -> jax.Array:
        """Observer-frame total merger rate, in mergers per second (scalar)."""

    def compute_population_terms(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> PopulationTerms:
        """Evaluate the population at a catalog's source parameters."""
        return PopulationTerms(
            log_prob=self.log_prob(source_parameters),
            log_luminosity_distance=jnp.log(
                self.luminosity_distance(source_parameters["redshift"])
            ),
            total_merger_rate=self.total_merger_rate(),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CosmologicalPopulation(Population):
    """Standard cosmological distance and a redshift-normalized merger rate.

    ``params["local_merger_rate"]`` is the local source-frame rate density in
    mergers per Gpc cubed per year. No modified-propagation parameters are
    required.
    """

    def luminosity_distance(self, redshift: ArrayLike) -> jax.Array:
        """Cosmological luminosity distance, in Mpc."""
        return self.redshift_distribution.luminosity_distance(redshift)

    def total_merger_rate(self) -> jax.Array:
        """Observer-frame total merger rate in mergers per second."""
        return self.redshift_distribution.total_merger_rate(
            self.params["local_merger_rate"]
        )


class PopulationTerms(NamedTuple):
    """A target population evaluated at a catalog's source parameters."""

    log_prob: jax.Array
    """Source log density, shape ``(N,)``."""

    log_luminosity_distance: jax.Array
    """Log distance governing waveform amplitude (distance in Mpc), shape ``(N,)``."""

    total_merger_rate: jax.Array
    """Observer-frame total merger rate in mergers per second (scalar)."""


def importance_log_weights(
    target: PopulationTerms,
    *,
    proposal_log_prob: jax.Array,
    log_reference_distance: jax.Array,
) -> jax.Array:
    """Source density ratio times inverse-square distance rescaling, in log space.

    The reference distance corresponds to the stored polarization power. No
    proposal merger rate is needed. Identical evaluated densities and distances
    give exactly zero log weights.
    """
    log_prob_ratio = target.log_prob - proposal_log_prob
    log_distance_ratio = target.log_luminosity_distance - log_reference_distance
    return log_prob_ratio - 2.0 * log_distance_ratio


class PopulationFn(Protocol):
    """Build a population at sampled hyperparameters, inside the sampler trace."""

    def __call__(self, params: Mapping[str, ArrayLike]) -> Population: ...


__all__ = [
    "CosmologicalPopulation",
    "Population",
    "PopulationFn",
    "PopulationTerms",
    "importance_log_weights",
]
