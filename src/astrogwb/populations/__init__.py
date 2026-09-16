"""Source populations declared as NumPyro models, addressed by registered name.

One declaration serves generation and inference: a population's source model
draws a catalog (:func:`~astrogwb.utils.sampling.sample_sources`), and
conditioning those draws back into it recovers the per-sample source density
the importance weights divide by
(:func:`~astrogwb.utils.sampling.evaluate_sources`). Its merger-rate function
returns the one observer-frame scalar the same catalog's Poisson count and
predicted spectrum need, or ``None`` when the population is a proposal density
with no physical rate. That is what makes a catalog self-describing -- the file
records one registry name, the construction kwargs the population was built
with, and the hyperparameters it was drawn at, which is everything needed to
reconstruct the density that produced it.

Importing this package registers every model it ships, the same way
:mod:`astrogwb.detector` exposes its own submodules. That import pulls in JAX
and NumPyro; :func:`astrogwb.paper.runtime.configure_runtime` must still run
before the XLA backend is initialized, but importing a model does not
initialize it.
"""

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
    DEFAULT_DENSITY_SITES,
    MergerRateFn,
    Population,
    SourceFn,
    build_population,
    known_populations,
    register_population,
)

__all__ = [
    "AMPLITUDE_PARAMETERS",
    "DEFAULT_DENSITY_SITES",
    "MergerRateFn",
    "Population",
    "SourceFn",
    "amplitude_H0_fn",
    "amplitude_local_merger_rate_fn",
    "bns_md_cosmological",
    "bns_md_gaussian_cosmological",
    "bns_md_gaussian_modified_propagation",
    "bns_md_gaussian_uniform_mixture",
    "bns_md_modified_propagation",
    "bns_md_uniform_mixture",
    "build_population",
    "known_populations",
    "madau_dickinson_total_merger_rate",
    "merger_rate_H0_fn",
    "merger_rate_local_merger_rate_fn",
    "register_population",
]
