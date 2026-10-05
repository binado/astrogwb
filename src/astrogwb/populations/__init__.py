"""Source populations declared as NumPyro models, addressed by registered name.

One declaration serves generation and inference: a population's source model
draws a catalog (:func:`~astrogwb.populations.evaluation.sample_sources`), and
conditioning those draws back into it recovers the per-sample source density
the importance weights divide by
(:func:`~astrogwb.populations.evaluation.evaluate_sources`). Its merger-rate function
returns the one observer-frame scalar the same catalog's Poisson count and
predicted spectrum need, or ``None`` when the population is a proposal density
with no physical rate. That is what makes a catalog self-describing -- the file
records one registry name, the registered redshift and mass sub-models and
construction kwargs the population was built with, and the hyperparameters it
was drawn at, which is everything needed to reconstruct the density that
produced it.

Importing this package registers every model it ships, the same way
:mod:`astrogwb.detector` exposes its own submodules. That import pulls in JAX
and NumPyro; :func:`astrogwb.paper.runtime.configure_runtime` must still run
before the XLA backend is initialized, but importing a model does not
initialize it.
"""

from astrogwb.populations.bns_madau_dickinson import (
    amplitude_H0_fn,
    amplitude_local_merger_rate_fn,
    bns_madau_dickinson,
    merger_rate_H0_fn,
    merger_rate_local_merger_rate_fn,
)
from astrogwb.populations.mass import (
    MassFn,
    build_mass_model,
    known_mass_models,
    register_mass_model,
)
from astrogwb.populations.metadata import ComponentMetadata, PopulationMetadata
from astrogwb.populations.orientation import IsotropicInclination
from astrogwb.populations.redshift import (
    RedshiftFn,
    RedshiftLaw,
    RedshiftModel,
    build_redshift_model,
    known_redshift_models,
    register_redshift_model,
)
from astrogwb.populations.registry import (
    DEFAULT_DENSITY_SITES,
    ComponentRegistry,
    MergerRateFn,
    Population,
    SourceFn,
    build_population,
    known_populations,
    register_population,
)

__all__ = [
    "DEFAULT_DENSITY_SITES",
    "ComponentMetadata",
    "ComponentRegistry",
    "IsotropicInclination",
    "MassFn",
    "MergerRateFn",
    "Population",
    "PopulationMetadata",
    "RedshiftFn",
    "RedshiftLaw",
    "RedshiftModel",
    "SourceFn",
    "amplitude_H0_fn",
    "amplitude_local_merger_rate_fn",
    "bns_madau_dickinson",
    "build_mass_model",
    "build_population",
    "build_redshift_model",
    "known_mass_models",
    "known_populations",
    "known_redshift_models",
    "merger_rate_H0_fn",
    "merger_rate_local_merger_rate_fn",
    "register_mass_model",
    "register_population",
    "register_redshift_model",
]
