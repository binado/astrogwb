"""Source populations declared as NumPyro models, addressed by registered name.

One declaration serves generation and inference: ``Predictive`` draws a catalog
from it, and substituting those draws back into it recovers the per-sample
source density the importance weights divide by. That is what makes a catalog
self-describing -- the file records a registry name, the model's construction
settings, and the hyperparameters it was drawn at, which is everything needed
to reconstruct the density that produced it.

Importing this package registers every model it ships, the same way
:mod:`astrogwb.detector` exposes its own submodules. That import pulls in JAX
and NumPyro; :func:`astrogwb.paper.runtime.configure_runtime` must still run
before the XLA backend is initialized, but importing a model does not
initialize it.
"""

from astrogwb.populations.bns_madau_dickinson import (
    AMPLITUDE_PARAMETERS,
    BNS_HIDDEN_SITES,
    amplitude_H0_fn,
    amplitude_local_merger_rate_fn,
    bns_md_cosmological,
    bns_md_modified_propagation,
    bns_md_uniform_mixture,
    merger_rate_H0_fn,
    merger_rate_local_merger_rate_fn,
)
from astrogwb.populations.evaluation import (
    LUMINOSITY_DISTANCE_SITE,
    REDSHIFT_SITE,
    TOTAL_MERGER_RATE_SITE,
    PopulationSites,
    PopulationTrace,
    included_log_prob,
    population_log_probs,
    population_sites,
    redshift_log_density,
    required_deterministic,
    select_stochastic_values,
)
from astrogwb.populations.generation import (
    derive_source_columns,
    draw_population,
)
from astrogwb.populations.registry import (
    PopulationModel,
    PopulationModelFn,
    known_population_models,
    population_model,
    register_population_model,
)

__all__ = [
    "AMPLITUDE_PARAMETERS",
    "BNS_HIDDEN_SITES",
    "LUMINOSITY_DISTANCE_SITE",
    "REDSHIFT_SITE",
    "TOTAL_MERGER_RATE_SITE",
    "PopulationModel",
    "PopulationModelFn",
    "PopulationSites",
    "PopulationTrace",
    "amplitude_H0_fn",
    "amplitude_local_merger_rate_fn",
    "bns_md_cosmological",
    "bns_md_modified_propagation",
    "bns_md_uniform_mixture",
    "derive_source_columns",
    "draw_population",
    "included_log_prob",
    "known_population_models",
    "merger_rate_H0_fn",
    "merger_rate_local_merger_rate_fn",
    "population_log_probs",
    "population_model",
    "population_sites",
    "redshift_log_density",
    "register_population_model",
    "required_deterministic",
    "select_stochastic_values",
]
