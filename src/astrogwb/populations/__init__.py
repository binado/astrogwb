"""Intrinsic source populations declared as NumPyro models.

A population's source model draws a catalog
(:func:`~astrogwb.utils.numpyro.sample_sources_qmc` or
:func:`~astrogwb.utils.numpyro.sample_and_logprob`), and conditioning those
draws back into it with :func:`~astrogwb.utils.numpyro.sample_and_logprob`
recovers the per-sample source density the importance weights divide by.
:mod:`astrogwb.populations.coba` is the BNS model. Callers supply the
redshift distribution and merger rate (:mod:`astrogwb.rates`).
"""

from astrogwb.populations import coba
from astrogwb.populations._types import Parameters, PopulationModel, SourceModel

__all__ = ["Parameters", "PopulationModel", "SourceModel", "coba"]
