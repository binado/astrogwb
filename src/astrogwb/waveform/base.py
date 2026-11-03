"""Common interface for waveform generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from astrogwb.waveform.metadata import WaveformMetadata

__all__ = ["WaveformGenerator"]


@dataclass(frozen=True, slots=True)
class WaveformGenerator:
    """Base for live generators; persisted description lives in ``metadata``."""

    metadata: WaveformMetadata

    @property
    def frequencies(self) -> ArrayLike:
        """The frequency axis produced by the concrete generator."""
        raise NotImplementedError

    def check_sources(self, source_parameters: Mapping[str, ArrayLike]) -> None:
        """Check eager source values supported by a concrete generator.

        Parameters
        ----------
        source_parameters
            Source parameter values by name.
        """
        del source_parameters

    def generate(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Generate power for one source.

        Parameters
        ----------
        source_parameters
            Source parameter values by name.
        """
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

        Trace-safe; an eager whole-catalog caller should ``jit`` it.
        ``chunk_size`` bounds peak memory via :func:`jax.lax.map`, changing the
        output only at rounding level. ``None`` generates one batch.

        Parameters
        ----------
        source_parameters
            Source parameter values by name.
        chunk_size
            Number of sources per chunk.
        """
        return _chunked(self._generate_batch, source_parameters, chunk_size)

    def generate_polarizations_batch(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        r"""Generate the complex polarizations of a catalog, each ``(N, F)``.

        :math:`(\tilde h_+, \tilde h_\times)` in each source's radiation frame:
        they depend on the intrinsic parameters, the inclination and the
        coalescence phase and optional geocentric arrival time, never on the
        sky position or the polarization angle,
        which enter through the antenna patterns. Trace-safe.

        Parameters
        ----------
        source_parameters
            Source parameter values by name. Optional ``coa_time`` gives
            the geocentric arrival time in seconds relative to the chosen origin,
            one per source, ``(N,)`` (or a scalar shared by all sources).
            Applies ``exp(-2 pi i f coa_time)`` to both polarizations.
            A missing column leaves them unshifted. Detector delays are applied by
            detector projection; power-only generation is independent of time.
        """
        plus, cross = self._generate_polarizations_batch(source_parameters)
        coa_time = source_parameters.get("coa_time")
        if coa_time is None:
            return plus, cross
        phase = jnp.exp(
            -2j
            * jnp.pi
            * jnp.asarray(coa_time)[..., None]
            * jnp.asarray(self.frequencies)
        )
        return plus * phase, cross * phase

    def _generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Every source in one vectorized pass; what a concrete generator implements."""
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
