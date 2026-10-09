"""Compose a population's two halves into the model a catalog is drawn from."""

from __future__ import annotations

import jax

from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.populations._types import PopulationModel, SourceModel

__all__ = ["joint_model"]


def joint_model(
    redshift_distribution: RedshiftDistribution, source_model: SourceModel
) -> PopulationModel:
    """The model that samples redshift, GW distance and the intrinsic sources.

    ``redshift_distribution.distance_model()`` contributes the ``redshift``
    site and the ``luminosity_distance`` deterministic; ``source_model`` the
    intrinsic sites. The returned mapping is the union of both, the columns a
    catalog stores. Call it as ``joint_model(*population(params))``.

    Parameters
    ----------
    redshift_distribution
        The population's redshift law, built at its hyperparameters.
    source_model
        The intrinsic model; it must not declare ``redshift`` or
        ``luminosity_distance``.
    """
    distance_model = redshift_distribution.distance_model()

    def model() -> dict[str, jax.Array]:
        return {**distance_model(), **source_model()}

    return model
