"""NumPyro ``Distribution`` classes for the population models.

Where :mod:`astrogwb.importance` packages a population as a closure returning
merger rate and log-weights, this subpackage expresses the same densities as
NumPyro distributions, so they can serve as latent sites, be sampled directly,
and carry their own cosmology tables.

Import from explicit submodules rather than this package root -- re-exporting
here would drag NumPyro into every importer of
:mod:`astrogwb.distributions.rates`, which is NumPyro-free precisely so the
reference callbacks can share it:

- :mod:`astrogwb.distributions.rates` — dimensionless merger-rate shapes
- :mod:`astrogwb.distributions.interpolated` — table-driven univariate density
- :mod:`astrogwb.distributions.redshift` — redshift densities from merger-rate
  models
"""

__all__: list[str] = []
