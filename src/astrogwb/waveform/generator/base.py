"""Common interface for polarization-power generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from astrogwb.metadata import WaveformMetadata

__all__ = ["PolarizationPowerGenerator"]


@dataclass(frozen=True, slots=True)
class PolarizationPowerGenerator:
    """Base for live generators; persisted description lives in ``metadata``."""

    metadata: WaveformMetadata

    @property
    def frequencies(self) -> ArrayLike:
        """The frequency axis produced by the concrete generator."""
        raise NotImplementedError

    def check_sources(self, source_parameters: Mapping[str, ArrayLike]) -> None:
        """Check eager source values supported by a concrete generator."""
        del source_parameters

    def generate(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power for one source."""
        power = self.generate_batch(
            {
                name: jnp.atleast_1d(jnp.asarray(value))
                for name, value in source_parameters.items()
            }
        )
        if power.shape[-1] != 1:
            raise ValueError(
                f"generate expects a single source; received {power.shape[-1]} events"
            )
        return power[:, 0]

    def generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate frequency-first power for a catalog."""
        raise NotImplementedError

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Return the generated frequency axis and power."""
        return jnp.asarray(self.frequencies), self.generate_batch(source_parameters)
