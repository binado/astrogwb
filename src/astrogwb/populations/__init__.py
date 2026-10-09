"""Source populations declared as NumPyro models, addressed by registered name.

One declaration serves generation and inference: a population's source model
draws a catalog (:func:`~astrogwb.populations.evaluation.sample_sources`), and
conditioning those draws back into it recovers the per-sample source density
the importance weights divide by
(:func:`~astrogwb.populations.evaluation.evaluate_sources`). A population also
returns its redshift distribution, whose total merger rate is the one
observer-frame scalar the same catalog's Poisson count and predicted spectrum
need, from the same call. That is what makes a
catalog self-describing -- the file records one registry name, the construction kwargs the population was built
with, and the hyperparameters it was drawn at, which is everything needed to
reconstruct the density that produced it.

Importing this package registers every model it ships, the same way
:mod:`astrogwb.detector` exposes its own submodules. That import pulls in JAX
and NumPyro; :func:`astrogwb.paper.runtime.configure_runtime` must still run
before the XLA backend is initialized, but importing a model does not
initialize it.
"""

from astrogwb.populations.bns_coba import bns_coba_population_fn
from astrogwb.populations.joint import joint_model
from astrogwb.populations.metadata import PopulationMetadata
from astrogwb.populations.registry import (
    DEFAULT_DENSITY_SITES,
    INTRINSIC_DENSITY_SITES,
    Population,
    build_population,
    known_populations,
    register_population,
)

__all__ = [
    "DEFAULT_DENSITY_SITES",
    "INTRINSIC_DENSITY_SITES",
    "Population",
    "PopulationMetadata",
    "bns_coba_population_fn",
    "build_population",
    "joint_model",
    "known_populations",
    "register_population",
]
