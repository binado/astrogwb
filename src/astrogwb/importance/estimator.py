"""Population-based importance estimator for the GWB spectral density."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.catalog.importance import ImportanceCatalog
from astrogwb.gwb.spectral import AverageMode, spectral_density
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.importance.population import PopulationFn, importance_log_weights


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpectralDensityImportanceEstimator:
    """Evaluate spectra from a fixed catalog and a ``params -> Population`` factory.

    The catalog is dynamic pytree data. The factory and inclination convention
    are static metadata; construct the factory once and reuse it. The factory
    must be hashable, and sampled parameters must arrive through its argument,
    rather than being captured in static metadata.

    ``__call__`` returns ``(spectrum, extras)`` with a fixed diagnostics key set,
    matching the spectral-density sampling protocol without depending on it.

    Construct the catalog outside the sampler, using the effective distances
    at which the power was computed, then reuse the estimator::

        catalog = ImportanceCatalog.from_population(
            population=population_fn(fiducials),
            source_parameters=samples,
            polarization_power=power,
            luminosity_distance=reference_distances,
        )
        estimator = SpectralDensityImportanceEstimator(
            catalog, population_fn, average_mode="analytic_inclination"
        )
        spectrum, extras = estimator(params)

    Fixed population-factory inputs, such as a redshift grid, can be bound once
    with ``functools.partial``. JIT and vmap operate on sampled parameters; to
    pass the catalog dynamically, use ``jax.jit(lambda e, p: e(p))``.
    """

    catalog: ImportanceCatalog
    population_fn: PopulationFn = field(metadata={"static": True})
    average_mode: AverageMode = field(metadata={"static": True})

    def __call__(
        self, params: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, Mapping[str, ArrayLike]]:
        """Return the spectrum, total merger rate, and relative importance ESS."""
        population = self.population_fn(params)
        target = population.compute_population_terms(self.catalog.source_parameters)
        log_weights = importance_log_weights(
            target,
            proposal_log_prob=self.catalog.proposal_log_prob,
            log_reference_distance=self.catalog.log_reference_distance,
        )
        prediction = spectral_density(
            self.catalog.polarization_power,
            jnp.exp(log_weights),
            target.total_merger_rate,
            average_mode=self.average_mode,
        )
        return prediction, {
            "total_merger_rate": target.total_merger_rate,
            "importance_relative_ess": relative_ess(log_weights),
        }


__all__ = ["SpectralDensityImportanceEstimator"]
