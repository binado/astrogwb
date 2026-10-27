"""Common interface for polarization-power generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
        whole-catalog caller should ``jit`` it. With ``chunk_size``, the full
        chunks are generated one batch at a time under :func:`jax.lax.map` and
        the remainder in one more batch, bounding peak memory to
        ``chunk_size`` sources' worth of waveform intermediates; it changes
        cost, and the output only at rounding level. ``None`` generates every
        source in one batch.
        """
        return _chunked(self._generate_batch, source_parameters, chunk_size)

    def generate_stokes(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate the Stokes parameters ``(4, F)`` of one source."""
        stokes = self.generate_stokes_batch(
            {
                name: jnp.atleast_1d(jnp.asarray(value))
                for name, value in source_parameters.items()
            }
        )
        if stokes.shape[-1] != 1:
            raise ValueError(
                "generate_stokes expects a single source; received "
                f"{stokes.shape[-1]} events"
            )
        return stokes[..., 0]

    def generate_stokes_batch(
        self,
        source_parameters: Mapping[str, ArrayLike],
        *,
        chunk_size: int | None = None,
    ) -> jax.Array:
        r"""Generate the Stokes parameters ``(4, F, N)`` of a catalog.

        :math:`(I, Q, U, V)` in each source's radiation frame at zero
        polarization angle (see :mod:`astrogwb.waveform.polarization_power`);
        :math:`I` is :meth:`generate_batch`'s power. Trace-safe, and
        ``chunk_size`` bounds peak memory as it does for :meth:`generate_batch`.
        """
        return _chunked(self._generate_stokes_batch, source_parameters, chunk_size)

    def generate_polarizations_batch(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        r"""Generate the complex polarizations of a catalog, each ``(N, F)``.

        :math:`(\tilde h_+, \tilde h_\times)` in each source's radiation frame:
        they depend on the intrinsic parameters, the inclination and the
        coalescence phase, never on the sky position or the polarization angle,
        which enter through the antenna patterns. Their
        :func:`~astrogwb.waveform.polarization_power.polarization_stokes` is
        :meth:`generate_stokes_batch`'s output. Trace-safe.
        """
        return self._generate_polarizations_batch(source_parameters)

    def _generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Every source in one vectorized pass; what a concrete generator implements."""
        del source_parameters
        raise NotImplementedError

    def _generate_stokes_batch(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> jax.Array:
        """The Stokes parameters of every source in one pass, ``(4, F, N)``."""
        del source_parameters
        raise NotImplementedError

    def _generate_polarizations_batch(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """The polarizations of every source in one pass, each ``(N, F)``."""
        del source_parameters
        raise NotImplementedError

    def __call__(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """Return the generated frequency axis and power."""
        return jnp.asarray(self.frequencies), self.generate_batch(source_parameters)


def _chunked(
    generate: Callable[[Mapping[str, ArrayLike]], jax.Array],
    source_parameters: Mapping[str, ArrayLike],
    chunk_size: int | None,
) -> jax.Array:
    """``generate`` over the catalog, whole or in chunks, sources on the last axis.

    The full chunks run under :func:`jax.lax.map` and the remainder as one more
    batch, so peak memory is ``chunk_size`` sources' worth of intermediates.
    ``generate`` returns ``(..., n)`` for ``n`` sources.
    """
    if chunk_size is None:
        return generate(source_parameters)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    columns = {name: jnp.asarray(value) for name, value in source_parameters.items()}
    count = next(iter(columns.values())).shape[0]
    full = count - count % chunk_size
    pieces = []
    if full:
        chunks = {
            name: value[:full].reshape(full // chunk_size, chunk_size)
            for name, value in columns.items()
        }
        # (n_chunks, ..., chunk_size) -> (..., full), sources in order.
        mapped = jax.lax.map(generate, chunks)
        ordered = jnp.moveaxis(mapped, 0, -2)
        pieces.append(ordered.reshape(*ordered.shape[:-2], full))
    if full < count:
        tail = {name: value[full:] for name, value in columns.items()}
        pieces.append(generate(tail))
    return jnp.concatenate(pieces, axis=-1)
