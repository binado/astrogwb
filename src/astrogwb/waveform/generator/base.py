"""Common interface for polarization-power generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from astrogwb.waveform.metadata import WaveformMetadata

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

    def generate_batch(
        self,
        source_parameters: Mapping[str, ArrayLike],
        *,
        chunk_size: int | None = None,
    ) -> jax.Array:
        """Generate frequency-first power ``(F, N)`` for a catalog.

        Trace-safe, so it composes inside ``jit`` and NumPyro models; an eager
        whole-catalog caller should ``jit`` it. ``chunk_size`` is forwarded to
        :func:`jax.lax.map` as ``batch_size``, bounding peak memory to
        ``chunk_size`` sources' worth of waveform intermediates; it changes
        cost, and the output only at rounding level. ``None`` generates every
        source in one vectorized pass.
        """
        if chunk_size is None:
            return self._generate_batch(source_parameters)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        columns = {
            name: jnp.asarray(value) for name, value in source_parameters.items()
        }

        def one(source: Mapping[str, jax.Array]) -> jax.Array:
            single = {name: value[None] for name, value in source.items()}
            return self._generate_batch(single)[:, 0]

        # ``lax.map`` vmaps ``one`` over ``chunk_size`` sources at a time and
        # handles the remainder itself.
        return jax.lax.map(one, columns, batch_size=chunk_size).T

    def _generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Every source in one vectorized pass; what a concrete generator implements."""
        raise NotImplementedError

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Return the generated frequency axis and power."""
        return jnp.asarray(self.frequencies), self.generate_batch(source_parameters)
