"""Source populations declared as NumPyro models, addressed by registered name.

One declaration serves generation and inference: a population's source model
draws a catalog (:func:`~astrogwb.populations.evaluation.sample_and_logprob`), and
conditioning those draws back into it recovers the per-sample source density
the importance weights divide by
(:func:`~astrogwb.populations.evaluation.sample_and_logprob`). A population also
returns its redshift distribution, whose total merger rate sets the catalog's
Poisson count and predicted spectrum. A catalog file therefore records only the
registry name, construction kwargs and hyperparameters needed to rebuild the
density that drew it.

Importing this package registers every model it ships.
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
