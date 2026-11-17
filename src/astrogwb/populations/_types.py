from collections.abc import Callable, Mapping
from typing import Protocol

import jax
from jax.typing import ArrayLike

type Parameters = Mapping[str, ArrayLike]

#: A no-argument NumPyro model declaring catalog columns as sample and deterministic sites.
type PopulationModel = Callable[[], None]

#: A no-argument NumPyro model of the intrinsic source parameters alone: it
#: knows nothing about redshift or distance.
type SourceModel = Callable[[], None]


class Population(Protocol):
    """A source-frame merger rate and intrinsic source model."""

    def merger_rate(self, redshift: ArrayLike, parameters: Parameters) -> jax.Array: ...

    def source_model(self, parameters: Parameters) -> SourceModel: ...
