"""Importance sampling of a fixed catalog against a target population.

:mod:`astrogwb.sampling.models` consumes any ``params -> (spectrum, extras)``
callable. This subpackage supplies the catalog-based realization of one: the
functions that reweight a fixed catalog to a target NumPyro source model, and
the effective-sample-size diagnostic that says how much of that catalog is
still doing work.

Import from explicit submodules rather than this package root:

- :mod:`astrogwb.populations` -- populations as NumPyro model declarations
- :mod:`astrogwb.catalog` -- the catalog that records the density that drew it
- :mod:`astrogwb.importance.spectral` -- catalog preparation and the weighted spectrum
- :mod:`astrogwb.importance.weights` -- the importance-weight arithmetic
- :mod:`astrogwb.importance.diagnostics` -- effective sample size helper
- :mod:`astrogwb.cosmology` -- shared cosmology helpers
"""

__all__: list[str] = []
