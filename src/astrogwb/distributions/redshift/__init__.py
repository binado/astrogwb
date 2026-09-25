"""Redshift distributions induced by source-frame merger-rate models.

The base, time-delayed and Madau-Dickinson implementations are also exported
here:

- :mod:`astrogwb.distributions.redshift.base` — the distribution and its
  cosmology tables
- :mod:`astrogwb.distributions.redshift.time_delayed` — mergers delayed from a
  formation rate
- :mod:`astrogwb.distributions.redshift.madau_dickinson` — the
  Madau-Dickinson rate-shape factories
"""

from astrogwb.distributions.redshift.base import (
    RedshiftDistribution,
    SourceFrameDistributionFn,
)
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
    madau_dickinson_redshift_distribution,
    madau_dickinson_time_delayed_redshift_distribution,
)
from astrogwb.distributions.redshift.time_delayed import (
    TimeDelayedRedshiftDistribution,
)

__all__: list[str] = [
    "MadauDickinsonRedshiftDistribution",
    "RedshiftDistribution",
    "SourceFrameDistributionFn",
    "TimeDelayedRedshiftDistribution",
    "madau_dickinson_redshift_distribution",
    "madau_dickinson_time_delayed_redshift_distribution",
]
