"""Redshift distributions induced by source-frame merger-rate models.

The base and Madau-Dickinson implementation are also exported here:

- :mod:`astrogwb.distributions.redshift.base` — the abstract base and its
  cosmology tables
- :mod:`astrogwb.distributions.redshift.madau_dickinson` — the
  Madau-Dickinson rate shape
"""

from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.distributions.redshift.madau_dickinson import (
    MadauDickinsonRedshiftDistribution,
)

__all__: list[str] = [
    "MadauDickinsonRedshiftDistribution",
    "RedshiftDistribution",
]
