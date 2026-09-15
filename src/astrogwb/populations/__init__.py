"""Source populations declared as NumPyro models, addressed by registered name.

One pair of declarations serves generation and inference: a source model
draws a catalog (:func:`~astrogwb.utils.sampling.sample_sources`), and
conditioning those draws back into it recovers the per-sample source density
the importance weights divide by
(:func:`~astrogwb.utils.sampling.evaluate_sources`). A
merger-rate function returns the one observer-frame scalar the same catalog's
Poisson count and predicted spectrum need. That is what makes a catalog
self-describing -- the file records the two registry names, both callables'
construction settings, and the hyperparameters they were drawn at, which is
everything needed to reconstruct the density that produced it.

The two declarations stay separate because they are independent: a
guard-mixture *redshift law* pairs with the same Madau-Dickinson *rate* the
physical population uses. Where a population's redshift law does determine its
own rate -- any model drawing redshift from a
:class:`~astrogwb.distributions.redshift.base.RedshiftDistribution` -- the rate
need not be restated by hand: :func:`infer_merger_rate_fn` derives it from the
declaration itself, and a core test holds the registered rate to the same
value.

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
from astrogwb.populations.merger_rate import (
    ABSOLUTE_RATE_PARAMETER,
    infer_merger_rate_fn,
    probe_redshift_distribution,
    require_absolute_rate,
)
from astrogwb.populations.record import PopulationRecord
from astrogwb.populations.registry import (
    DEFAULT_DENSITY_SITES,
    DEFAULT_MERGER_RATE_MODEL,
    REDSHIFT_SITE,
    SHARED_MODEL_KWARGS,
    MergerRateFn,
    SourceFn,
    build_merger_rate_fn,
    build_source_model,
    known_merger_rate_models,
    known_source_models,
    register_merger_rate_model,
    register_source_model,
)

__all__ = [
    "ABSOLUTE_RATE_PARAMETER",
    "AMPLITUDE_PARAMETERS",
    "DEFAULT_DENSITY_SITES",
    "DEFAULT_MERGER_RATE_MODEL",
    "REDSHIFT_SITE",
    "SHARED_MODEL_KWARGS",
    "MergerRateFn",
    "PopulationRecord",
    "SourceFn",
    "amplitude_H0_fn",
    "amplitude_local_merger_rate_fn",
    "bns_md_cosmological",
    "bns_md_gaussian_cosmological",
    "bns_md_gaussian_modified_propagation",
    "bns_md_gaussian_uniform_mixture",
    "bns_md_modified_propagation",
    "bns_md_uniform_mixture",
    "build_merger_rate_fn",
    "build_source_model",
    "infer_merger_rate_fn",
    "known_merger_rate_models",
    "known_source_models",
    "madau_dickinson_total_merger_rate",
    "merger_rate_H0_fn",
    "merger_rate_local_merger_rate_fn",
    "probe_redshift_distribution",
    "register_merger_rate_model",
    "register_source_model",
    "require_absolute_rate",
]
