"""Common interface for waveform generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.waveform.metadata import WaveformConfig

__all__ = ["WaveformGenerator"]


@dataclass(frozen=True, slots=True)
class WaveformGenerator:
    """Base for live generators; persisted description lives in ``metadata``."""

    metadata: WaveformConfig

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

    def intensity(
        self,
        sources: Mapping[str, ArrayLike],
        *,
        chunk_size: int | None = None,
    ) -> jax.Array:
        """Generate frequency-domain intensity ``(N, F)`` for a catalog.

        Trace-safe; an eager whole-catalog caller should ``jit`` it.
        ``chunk_size`` bounds peak memory via :func:`jax.lax.map`, changing the
        output only at rounding level. ``None`` generates one batch.

        Parameters
        ----------
        sources
            Source parameter values by name.
        chunk_size
            Number of sources per chunk.
        """
        return _chunked(self._generate_batch, sources, chunk_size)

    def polarizations(
        self, sources: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        r"""Generate the complex polarizations of a catalog, each ``(N, F)``.

        :math:`(\tilde h_+, \tilde h_\times)` in each source's radiation frame:
        they depend on the intrinsic parameters, the inclination and the
        coalescence phase and optional geocentric arrival time, never on the
        sky position or the polarization angle,
        which enter through the antenna patterns. Trace-safe.

        Parameters
        ----------
        sources
            Source parameter values by name. Optional ``coa_time`` gives
            the geocentric arrival time in seconds relative to the chosen origin,
            one per source, ``(N,)`` (or a scalar shared by all sources).
            Applies ``exp(-2 pi i f coa_time)`` to both polarizations.
            A missing column leaves them unshifted. Detector delays are applied by
            detector projection; intensity-only generation is independent of time.
        """
        plus, cross = self._generate_polarizations_batch(sources)
        coa_time = sources.get("coa_time")
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


def _chunked(
    generate: Callable[[Mapping[str, ArrayLike]], jax.Array],
    source_parameters: Mapping[str, ArrayLike],
    chunk_size: int | None,
) -> jax.Array:
    """``generate`` over the catalog, whole or in chunks, sources on the first axis.

    The full chunks run under :func:`jax.lax.map` and the remainder as one more
    batch, so peak memory is ``chunk_size`` sources' worth of intermediates.
    ``generate`` returns ``(n, F)`` for ``n`` sources.
    """
    if chunk_size is None:
        return generate(source_parameters)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    columns = {name: jnp.asarray(value) for name, value in source_parameters.items()}
    count = next(iter(columns.values())).shape[0]
    full = count - count % chunk_size
    pieces = []
    if count == 0:
        return generate(columns)
    if full:
        chunks = {
            name: value[:full].reshape(full // chunk_size, chunk_size)
            for name, value in columns.items()
        }
        mapped = jax.lax.map(generate, chunks)
        pieces.append(mapped.reshape(full, mapped.shape[-1]))
    if full < count:
        tail = {name: value[full:] for name, value in columns.items()}
        pieces.append(generate(tail))
    return jnp.concatenate(pieces, axis=0)
