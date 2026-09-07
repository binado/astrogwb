"""Importance-sampling reference models and cosmology helpers for the SGWB.

:mod:`astrogwb.sampling.models` consumes any ``params -> (spectrum, extras)``
callable. This subpackage supplies the population-based realization of one:
populations as pytrees of distributions, the weights between two of them, and
the estimator that contracts a fixed catalog against them.

Import from explicit submodules rather than this package root:

- :mod:`astrogwb.cosmology` — shared cosmology helpers
- :mod:`astrogwb.population` — populations as pytrees of distributions
- :mod:`astrogwb.importance.weights` — importance weights between populations
- :mod:`astrogwb.importance.estimator` — population-based spectral estimator
- :mod:`astrogwb.importance.diagnostics` — effective sample size helper
- :mod:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation` —
  the BNS + Madau-Dickinson reference population
"""

__all__: list[str] = []
