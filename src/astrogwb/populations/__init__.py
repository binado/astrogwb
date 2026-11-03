"""Source populations declared as NumPyro models, described by typed records.

One declaration serves generation and inference: a population's source model
draws a catalog (:func:`~astrogwb.utils.numpyro.sample_and_logprob`), and
conditioning those draws back into it recovers the per-sample source density
the importance weights divide by
(:func:`~astrogwb.utils.numpyro.sample_and_logprob`). A population also
returns its redshift distribution, whose total merger rate sets the catalog's
Poisson count and predicted spectrum.

Each population is a :class:`PopulationMetadata` subclass whose fields are its
construction settings; :data:`AnyPopulation` deserializes a record to the
subclass its ``model_name`` names. A catalog file therefore records only that
record and the hyperparameters needed to rebuild the density that drew it.
"""

from astrogwb.populations._types import Population
from astrogwb.populations.base import PopulationMetadata
from astrogwb.populations.bns_coba import BNSCobaPopulation
from astrogwb.populations.joint import (
    DEFAULT_DENSITY_SITES,
    INTRINSIC_DENSITY_SITES,
    REDSHIFT_SITE,
    joint_model,
)
from astrogwb.populations.metadata import AnyPopulation

__all__ = [
    "DEFAULT_DENSITY_SITES",
    "INTRINSIC_DENSITY_SITES",
    "REDSHIFT_SITE",
    "AnyPopulation",
    "BNSCobaPopulation",
    "Population",
    "PopulationMetadata",
    "joint_model",
]
