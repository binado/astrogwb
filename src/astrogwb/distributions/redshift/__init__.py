"""Redshift distributions induced by source-frame merger-rate models.

The base and Madau-Dickinson implementation are also exported here:

- :mod:`astrogwb.distributions.redshift.base` — the distribution and its
  cosmology tables
- :mod:`astrogwb.distributions.redshift.madau_dickinson` — the
  Madau-Dickinson rate-shape factory
"""

from astrogwb.distributions.redshift.base import (
    RedshiftDistribution,
    SourceFrameDistributionFn,
)
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
    madau_dickinson_redshift_distribution,
)

__all__: list[str] = [
    "MadauDickinsonRedshiftDistribution",
    "RedshiftDistribution",
    "SourceFrameDistributionFn",
    "madau_dickinson_redshift_distribution",
]
