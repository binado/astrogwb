"""Source populations declared as NumPyro models, addressed by registered name.

One pair of declarations serves generation and inference: a source model's
``Predictive`` draws a catalog, and substituting those draws back into it
recovers the per-sample source density the importance weights divide by. A
merger-rate model returns the one observer-frame scalar the same catalog's
Poisson count and predicted spectrum need. That is what makes a catalog
self-describing -- the file records the two registry names, both models'
construction settings, and the hyperparameters they were drawn at, which is
everything needed to reconstruct the density that produced it.

Importing this package registers every model it ships, the same way
:mod:`astrogwb.detector` exposes its own submodules. That import pulls in JAX
and NumPyro; :func:`astrogwb.paper.runtime.configure_runtime` must still run
before the XLA backend is initialized, but importing a model does not
initialize it.
"""

from astrogwb.populations.base import (
    MergerRateFn,
    MergerRateModel,
    Population,
    PopulationDraw,
    PopulationEvaluation,
    SourceEvaluation,
    SourceFn,
    SourceModel,
)
from astrogwb.populations.bns_madau_dickinson import (
    AMPLITUDE_PARAMETERS,
    amplitude_H0_fn,
    amplitude_local_merger_rate_fn,
    bns_md_cosmological,
    bns_md_gaussian_cosmological,
    bns_md_gaussian_modified_propagation,
    bns_md_gaussian_uniform_mixture,
    bns_md_modified_propagation,
    bns_md_uniform_mixture,
    madau_dickinson_total_merger_rate,
    merger_rate_H0_fn,
    merger_rate_local_merger_rate_fn,
)
from astrogwb.populations.registry import (
    SHARED_MODEL_KWARGS,
    build_merger_rate_model,
    build_population,
    build_source_model,
    known_merger_rate_models,
    known_source_models,
    recipe_name,
    register_merger_rate_model,
    register_recipe,
    register_source_model,
    resolve_recipe,
)

__all__ = [
    "AMPLITUDE_PARAMETERS",
    "SHARED_MODEL_KWARGS",
    "MergerRateFn",
    "MergerRateModel",
    "Population",
    "PopulationDraw",
    "PopulationEvaluation",
    "SourceEvaluation",
    "SourceFn",
    "SourceModel",
    "amplitude_H0_fn",
    "amplitude_local_merger_rate_fn",
    "bns_md_cosmological",
    "bns_md_gaussian_cosmological",
    "bns_md_gaussian_modified_propagation",
    "bns_md_gaussian_uniform_mixture",
    "bns_md_modified_propagation",
    "bns_md_uniform_mixture",
    "build_merger_rate_model",
    "build_population",
    "build_source_model",
    "known_merger_rate_models",
    "known_source_models",
    "madau_dickinson_total_merger_rate",
    "merger_rate_H0_fn",
    "merger_rate_local_merger_rate_fn",
    "recipe_name",
    "register_merger_rate_model",
    "register_recipe",
    "register_source_model",
    "resolve_recipe",
]
