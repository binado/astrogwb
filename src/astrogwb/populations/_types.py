from collections.abc import Callable, Mapping

import jax
from jax.typing import ArrayLike

type Parameters = Mapping[str, ArrayLike]
type PopulationModel = Callable[[], dict[str, jax.Array]]
type Population = Callable[[Parameters], tuple[jax.Array, PopulationModel]]
