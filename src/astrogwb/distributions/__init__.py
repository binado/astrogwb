"""NumPyro ``Distribution`` classes for the population models.

- :mod:`astrogwb.distributions.rates` — dimensionless merger-rate shapes
- :mod:`astrogwb.distributions.interpolated` — table-driven univariate density
- :mod:`astrogwb.distributions.redshift` — redshift densities from merger-rate
  models
- :mod:`astrogwb.distributions.mass` — ordered component-mass densities
- :mod:`astrogwb.distributions.orientation` — isotropic polar-angle densities
- :mod:`astrogwb.distributions.time_delay` — formation-to-merger delay densities
- :mod:`astrogwb.distributions.amplitude` — conditional posterior of a
  marginalized amplitude parameter
"""

from astrogwb.distributions.mass import MaxOfTwoNormalsDistribution
from astrogwb.distributions.orientation import UniformCosineDistribution
from astrogwb.distributions.time_delay import PowerLawTimeDelayDistribution

__all__ = [
    "MaxOfTwoNormalsDistribution",
    "PowerLawTimeDelayDistribution",
    "UniformCosineDistribution",
]
