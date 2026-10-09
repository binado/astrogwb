"""Diagnostics for importance sampling a fixed catalog against a target population.

The catalog-based spectrum itself is :mod:`astrogwb.gwb.importance`, which
reweights the intrinsic draws of a reference catalog and integrates redshift by
quadrature. This subpackage holds the effective-sample-size diagnostic that says
how much of that catalog is still doing work.

Import from explicit submodules rather than this package root:

- :mod:`astrogwb.gwb.importance` -- the rescaled quadrature spectrum and its weights
- :mod:`astrogwb.populations` -- populations as NumPyro model declarations
- :mod:`astrogwb.importance.diagnostics` -- effective sample size helper
- :mod:`astrogwb.cosmology` -- shared cosmology helpers
"""

__all__: list[str] = []
