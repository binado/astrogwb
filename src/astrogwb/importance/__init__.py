"""Importance sampling of a fixed catalog against a target population.

:mod:`astrogwb.sampling.models` consumes any ``params -> (spectrum, extras)``
callable. This subpackage supplies the population-based realization of one: the
estimator that contracts a fixed catalog against a NumPyro population model,
and the effective-sample-size diagnostic that says how much of that catalog is
still doing work.

Import from explicit submodules rather than this package root:

- :mod:`astrogwb.populations` -- populations as NumPyro model declarations
- :mod:`astrogwb.catalog` -- the catalog that records the density that drew it
- :mod:`astrogwb.importance.estimator` -- population-based spectral estimator
- :mod:`astrogwb.importance.diagnostics` -- effective sample size helper
- :mod:`astrogwb.cosmology` -- shared cosmology helpers
"""

__all__: list[str] = []
