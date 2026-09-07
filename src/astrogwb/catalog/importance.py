"""Fixed samples and reference quantities for spectral importance sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from astrogwb.population import Population


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ImportanceCatalog:
    """A fixed Monte Carlo realization and the quantities needed to reweight it.

    ``polarization_power`` has shape ``(F, N)``; source parameters and cached
    proposal quantities have shape ``(N,)``. All fields are dynamic pytree data.
    Treat the source mapping as immutable after construction.

    ``log_reference_distance`` is the log of the effective distance in Mpc
    governing the stored power's amplitude. It already includes any fiducial
    propagation modification, so it must not receive another one. A catalog
    whose stored source distances are the *EM* ones -- because the fiducial
    propagation was applied to the power instead -- must convert them before
    they reach this field.

    The constructor performs no validation or array conversion: JAX rebuilds
    instances internally while flattening and unflattening pytrees, possibly
    with tracers, placeholders, or additional batch dimensions.
    :meth:`from_population` only converts inputs to arrays and caches the
    proposal density and reference distances. Direct construction with
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

        Call outside JAX transformations: distances are supplied explicitly
        rather than recomputed from the population's cosmology. Inputs are
        converted to arrays without validation; ensuring consistent shapes
        and positive, finite linear distances is the caller's responsibility.
        """
        power = jnp.asarray(polarization_power)

        parameters = {
            name: jnp.asarray(values) for name, values in source_parameters.items()
        }

        distance = jnp.asarray(luminosity_distance)

        return cls(
            source_parameters=parameters,
            polarization_power=power,
            proposal_log_prob=population.log_prob(parameters),
            log_reference_distance=jnp.log(distance),
        )


__all__ = ["ImportanceCatalog"]
