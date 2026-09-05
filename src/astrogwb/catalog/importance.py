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


@jax.tree_util.register_pytree_node_class
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

    The ordinary constructor accepts precomputed proposal densities, including
    mixtures. The proposal need not itself be a physical population, and no
    proposal merger rate or observation time enters the importance estimator.
    """

    source_parameters: Mapping[str, jax.Array]
    polarization_power: jax.Array
    proposal_log_prob: jax.Array
    log_reference_distance: jax.Array

    def __post_init__(self) -> None:
        power = jnp.asarray(self.polarization_power)
        if power.ndim != 2 or not (
            jnp.issubdtype(power.dtype, jnp.floating)
            or jnp.issubdtype(power.dtype, jnp.integer)
        ):
            raise ValueError("polarization_power must be a real two-dimensional array")
        num_samples = power.shape[1]
        if num_samples == 0:
            raise ValueError("ImportanceCatalog requires at least one source")
        if "redshift" not in self.source_parameters:
            raise ValueError("ImportanceCatalog requires redshift samples")

        parameters = {}
        for name, values in self.source_parameters.items():
            if not isinstance(name, str):
                raise TypeError("source parameter names must be strings")
            array = jnp.asarray(values)
            if array.shape != (num_samples,):
                raise ValueError(
                    f"source parameter {name!r} must have shape ({num_samples},)"
                )
            parameters[name] = array

        object.__setattr__(self, "source_parameters", parameters)
        object.__setattr__(self, "polarization_power", power)
        for name in ("proposal_log_prob", "log_reference_distance"):
            array = jnp.asarray(getattr(self, name))
            if array.shape != (num_samples,):
                raise ValueError(f"{name} must have shape ({num_samples},)")
            if not (
                jnp.issubdtype(array.dtype, jnp.floating)
                or jnp.issubdtype(array.dtype, jnp.integer)
            ):
                raise ValueError(f"{name} must be real-valued")
            object.__setattr__(self, name, array)

    @classmethod
    def from_population(
        cls,
        *,
        population: Population,
        source_parameters: Mapping[str, ArrayLike],
        polarization_power: ArrayLike,
        luminosity_distance: ArrayLike,
    ) -> Self:
        """Cache proposal density and supplied effective reference distances once.

        Call outside JAX transformations: positive, finite linear distances
        are validated on the host. Distances are supplied explicitly rather
        than recomputed from the population's cosmology.
        """
        distance = np.asarray(luminosity_distance)
        if not np.isrealobj(distance) or not np.all(
            np.isfinite(distance) & (distance > 0)
        ):
            raise ValueError("luminosity_distance must be positive and finite")
        parameters = {
            name: jnp.asarray(values) for name, values in source_parameters.items()
        }
        if "redshift" not in parameters:
            raise ValueError("ImportanceCatalog requires redshift samples")
        return cls(
            source_parameters=parameters,
            polarization_power=jnp.asarray(polarization_power),
            proposal_log_prob=population.log_prob(parameters),
            log_reference_distance=jnp.log(jnp.asarray(distance)),
        )

    def tree_flatten(
        self,
    ) -> tuple[tuple[Mapping[str, jax.Array], jax.Array, jax.Array, jax.Array], None]:
        """Expose arrays as dynamic leaves with no static catalog metadata."""
        return (
            self.source_parameters,
            self.polarization_power,
            self.proposal_log_prob,
            self.log_reference_distance,
        ), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Mapping[str, jax.Array], jax.Array, jax.Array, jax.Array],
    ) -> Self:
        """Reconstruct transformed data without validation or host conversion.

        JAX may supply tracers, placeholders, or additional batch dimensions.
        The original user-facing construction already validated the catalog.
        """
        catalog = object.__new__(cls)
        for name, value in zip(
            (
                "source_parameters",
                "polarization_power",
                "proposal_log_prob",
                "log_reference_distance",
            ),
            children,
            strict=True,
        ):
            object.__setattr__(catalog, name, value)
        return catalog


__all__ = ["ImportanceCatalog"]
