from collections.abc import Callable, Mapping

import jax
from jax.typing import ArrayLike

from astrogwb.distributions.redshift.base import RedshiftDistribution

type Parameters = Mapping[str, ArrayLike]

#: A no-argument NumPyro model returning the columns a catalog stores.
type PopulationModel = Callable[[], dict[str, jax.Array]]

#: A no-argument NumPyro model of the intrinsic source parameters alone: it
#: knows nothing about redshift or distance.
type SourceModel = Callable[[], dict[str, jax.Array]]

#: ``parameters -> (redshift_distribution, source_model)``.
type Population = Callable[[Parameters], tuple[RedshiftDistribution, SourceModel]]
