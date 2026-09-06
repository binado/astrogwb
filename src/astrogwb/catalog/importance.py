"""Fixed samples and reference quantities for spectral importance sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from astrogwb.importance.population import Population


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ImportanceCatalog:
    """A fixed Monte Carlo realization and the quantities needed to reweight it.

    ``polarization_power`` has shape ``(F, N)``; source parameters and cached
    proposal quantities have shape ``(N,)``. All fields are dynamic pytree data.
    Treat the source mapping as immutable after construction.

    ``log_reference_distance`` is the log of the effective distance in Mpc
    governing the stored power's amplitude. It already includes any fiducial
    propagation modification. Unlike the legacy BNS callback's stored EM
    distances, it must not receive another propagation correction.

    The constructor performs no validation or array conversion: JAX rebuilds
    instances internally while flattening and unflattening pytrees, possibly
    with tracers, placeholders, or additional batch dimensions. Validated
    construction is :meth:`from_population`. Direct construction with
    precomputed proposal densities (including mixtures) remains supported; the
    proposal need not itself be a physical population, and no proposal merger
    rate or observation time enters the importance estimator. Ensuring
    consistent shapes is then the caller's responsibility.
    """

    source_parameters: Mapping[str, jax.Array]
    polarization_power: jax.Array
    proposal_log_prob: jax.Array
    log_reference_distance: jax.Array

    @classmethod
    def from_population(
        cls,
        *,
        population: Population,
        source_parameters: Mapping[str, ArrayLike],
        polarization_power: ArrayLike,
        luminosity_distance: ArrayLike,
    ) -> Self:
        """Cache the proposal density and supplied effective distances once.

        Call outside JAX transformations: array shapes and positive, finite
        linear distances are validated on the host. Distances are supplied
        explicitly rather than recomputed from the population's cosmology.
        """
        power = jnp.asarray(polarization_power)
        if power.ndim != 2 or not (
            np.issubdtype(power.dtype, np.floating)
            or np.issubdtype(power.dtype, np.integer)
        ):
            raise ValueError("polarization_power must be a real two-dimensional array")
        num_samples = power.shape[1]
        if num_samples == 0:
            raise ValueError("ImportanceCatalog requires at least one source")

        parameters = {
            name: jnp.asarray(values) for name, values in source_parameters.items()
        }
        if "redshift" not in parameters:
            raise ValueError("ImportanceCatalog requires redshift samples")
        for name, values in parameters.items():
            if values.shape != (num_samples,):
                raise ValueError(
                    f"source parameter {name!r} must have shape ({num_samples},)"
                )

        distance = np.asarray(luminosity_distance)
        if distance.shape != (num_samples,):
            raise ValueError(
                f"log_reference_distance requires distances of shape ({num_samples},)"
            )
        if not np.isrealobj(distance) or not np.all(
            np.isfinite(distance) & (distance > 0)
        ):
            raise ValueError("luminosity_distance must be positive and finite")

        return cls(
            source_parameters=parameters,
            polarization_power=jnp.asarray(power),
            proposal_log_prob=population.log_prob(parameters),
            log_reference_distance=jnp.log(jnp.asarray(distance)),
        )


__all__ = ["ImportanceCatalog"]
