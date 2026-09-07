r"""Source-population models, native sampling, and gwmock graph adaptation.

A population describes the source density at one hyperparameter point. It
also supplies the luminosity distance governing signal amplitude and the
observer-frame total merger rate. Concrete subclasses own the physics of
those two quantities; the generic population has no propagation parameters.

Population instances contain distributions and hyperparameters, never catalog
arrays. Concrete dataclass subclasses must each be registered as JAX pytrees
so their distribution and parameter values remain dynamic under transforms.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.distributions.redshift import RedshiftDistribution
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
)


def _concrete_scalar(value: ArrayLike, *, label: str) -> int | float:
    """Return one host scalar, failing clearly for traced or non-scalar values."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must be a concrete scalar for gwmock graph conversion"
        ) from error
    if array.shape != ():
        raise ValueError(
            f"{label} must be scalar for gwmock graph conversion, got shape "
            f"{array.shape}"
        )
    return array.item()


def _gwmock_normal(
    *, key: jax.Array, n_samples: int, loc: ArrayLike, scale: ArrayLike
) -> jax.Array:
    """Draw a normal sample using gwmock's graph-sampler calling convention."""
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}")
    if float(np.asarray(scale)) <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    return jnp.asarray(loc) + jnp.asarray(scale) * jax.random.normal(
        key, shape=(n_samples,)
    )


@dataclass(frozen=True)
class Population(ABC):
    r"""Source model :math:`p(x \mid \theta)` with distance and rate behavior.

    ``distributions`` must contain a ``RedshiftDistribution`` under
    ``redshift``. Other independent source-parameter laws may be supplied by
    name. ``params`` carries the hyperparameters needed by the concrete
    population; the base class requires no particular parameter names. Treat
    both mappings as immutable after construction.

    Only source densities that differ between target and proposal need to be
    included, except redshift, which is always required. Both populations must
    use the same source-parameter factors. Evaluate importance weights only
    where the proposal has support: subtracting two negative-infinite log
    densities produces ``nan``.
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

    def sample(
        self, key: jax.Array, sample_shape: tuple[int, ...] = ()
    ) -> dict[str, jax.Array]:
        """Draw every independent source parameter with its own split key."""
        keys = jax.random.split(key, len(self.distributions))
        return {
            name: jnp.asarray(distribution.sample(parameter_key, sample_shape))
            for (name, distribution), parameter_key in zip(
                self.distributions.items(), keys, strict=True
            )
        }

    def _to_gwmock_population_graph(self) -> dict[str, dict[str, Any]]:
        """Convert supported independent distributions to gwmock graph nodes.

        This host-side adapter requires concrete scalar distribution parameters.
        It intentionally models sampler nodes only; correlated draws and
        deterministic transforms remain the responsibility of explicit gwmock
        graph configurations.
        """
        graph: dict[str, dict[str, Any]] = {}
        for name, distribution in self.distributions.items():
            if distribution.batch_shape or distribution.event_shape:
                raise ValueError(
                    f"distribution {name!r} must have scalar batch and event shapes "
                    f"for gwmock graph conversion, got batch_shape="
                    f"{distribution.batch_shape} and event_shape="
                    f"{distribution.event_shape}"
                )

            if isinstance(distribution, MadauDickinsonRedshiftDistribution):
                parameter_names = ("H0", "Omega_m", "gamma", "kappa", "z_peak")
                missing = [key for key in parameter_names if key not in self.params]
                if missing:
                    raise ValueError(
                        f"distribution {name!r} requires Population.params keys "
                        f"{missing} for gwmock graph conversion"
                    )
                arguments = {
                    "z_min": _concrete_scalar(
                        distribution.minimum_redshift,
                        label=f"distribution {name!r} minimum_redshift",
                    ),
                    "z_max": _concrete_scalar(
                        distribution.maximum_redshift,
                        label=f"distribution {name!r} maximum_redshift",
                    ),
                    "gamma": _concrete_scalar(
                        self.params["gamma"], label="Population.params['gamma']"
                    ),
                    "kappa": _concrete_scalar(
                        self.params["kappa"], label="Population.params['kappa']"
                    ),
                    "z_peak": _concrete_scalar(
                        self.params["z_peak"], label="Population.params['z_peak']"
                    ),
                    "hubble_constant": _concrete_scalar(
                        self.params["H0"], label="Population.params['H0']"
                    ),
                    "omega_m": _concrete_scalar(
                        self.params["Omega_m"], label="Population.params['Omega_m']"
                    ),
                    "n_grid": distribution.n_grid,
                }
                function = "madau_dickinson_redshift"
            elif isinstance(distribution, dist.Uniform):
                arguments = {
                    "minimum": _concrete_scalar(
                        distribution.low, label=f"distribution {name!r} low"
                    ),
                    "maximum": _concrete_scalar(
                        distribution.high, label=f"distribution {name!r} high"
                    ),
                }
                function = "uniform"
            elif isinstance(distribution, dist.Normal):
                arguments = {
                    "loc": _concrete_scalar(
                        distribution.loc, label=f"distribution {name!r} loc"
                    ),
                    "scale": _concrete_scalar(
                        distribution.scale, label=f"distribution {name!r} scale"
                    ),
                }
                function = "astrogwb.population._gwmock_normal"
            else:
                raise TypeError(
                    f"distribution {name!r} has unsupported type "
                    f"{type(distribution).__name__}; gwmock graph conversion supports "
                    "Uniform, Normal, and MadauDickinsonRedshiftDistribution"
                )

            graph[name] = {"sampler": {"function": function, "arguments": arguments}}
        return graph

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
    """Log distance governing waveform amplitude, shape ``(N,)``."""

    total_merger_rate: jax.Array
    """Observer-frame total merger rate in mergers per second (scalar)."""


class PopulationFn(Protocol):
    """Build a population at sampled hyperparameters, inside the sampler trace."""

    def __call__(self, params: Mapping[str, ArrayLike]) -> Population: ...


__all__ = [
    "CosmologicalPopulation",
    "Population",
    "PopulationFn",
    "PopulationTerms",
]
