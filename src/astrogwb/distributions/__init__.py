"""NumPyro ``Distribution`` classes for the population models.

- :mod:`astrogwb.distributions.rates` — dimensionless merger-rate shapes
- :mod:`astrogwb.distributions.interpolated` — table-driven univariate density
- :mod:`astrogwb.distributions.redshift` — redshift densities from merger-rate
  models
- :mod:`astrogwb.distributions.mass` — ordered component-mass densities
- :mod:`astrogwb.distributions.amplitude` — conditional posterior of a
  marginalized amplitude parameter
"""

from astrogwb.distributions.mass import MaxOfTwoNormals

__all__ = ["MaxOfTwoNormals"]
