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
    """A source-frame merger rate and intrinsic source model.

    Both methods take the same hyperparameter mapping. Simulation builds a
    redshift distribution from ``merger_rate`` and composes it with
    ``source_model``; importance integration evaluates ``merger_rate`` pointwise
    and conditions ``source_model`` on catalog draws.
    """

    def merger_rate(self, redshift: ArrayLike, parameters: Parameters) -> jax.Array:
        """Source-frame merger rate density.

        Parameters
        ----------
        redshift
            Redshift, of any shape.
        parameters
            Hyperparameter values. Must carry whatever the rate reads.

        Returns
        -------
        jax.Array
            The rate in :math:`\\mathrm{Gpc}^{-3}\\,\\mathrm{yr}^{-1}` (a
            dimensionless shape when the population has no local rate), with
            the shape of ``redshift``.
        """
        ...

    def source_model(self, parameters: Parameters) -> SourceModel:
        """A no-argument NumPyro model of the intrinsic source parameters.

        It declares intrinsic sites only: never redshift, luminosity distance
        or extrinsic angles. Importance weights rely on that independence.

        Parameters
        ----------
        parameters
            Hyperparameter values.
        """
        ...
