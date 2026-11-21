"""Common interface for waveform generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.waveform.metadata import WaveformConfig

__all__ = ["WaveformGenerator"]


@dataclass(frozen=True, slots=True)
class WaveformGenerator(ABC):
    """Base for live generators; the persisted description lives in ``config``."""

    config: WaveformConfig

    @property
    @abstractmethod
    def frequencies(self) -> jax.Array:
        """The frequency axis in Hz, ``(F,)``, produced by the concrete generator."""

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
        """Generate frequency-domain intensity of a catalog of ``N`` sources.

        Trace-safe; an eager whole-catalog caller should ``jit`` it.
        ``chunk_size`` bounds peak memory via :func:`jax.lax.map`, changing the
        output only at rounding level. ``None`` generates one batch.

        Parameters
        ----------
        sources
            Source parameter values by name, each ``(N,)``.
        chunk_size
            Number of sources per chunk. It fixes array shapes, so it must be a
            static argument when the method is jit compiled.

        Returns
        -------
        jax.Array
            The intensity, ``(N, F)``, with sources on the first axis.

        Raises
        ------
        ValueError
            If ``chunk_size`` is not positive, or the columns of ``sources`` are
            missing or differ in length.
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

        Returns
        -------
        tuple of jax.Array
            :math:`\tilde h_+` and :math:`\tilde h_\times`, each ``(N, F)``.
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

    @abstractmethod
    def _generate_batch(self, source_parameters: Mapping[str, ArrayLike]) -> jax.Array:
        """Every source in one vectorized pass, ``(N, F)``."""

    @abstractmethod
    def _generate_polarizations_batch(
        self, source_parameters: Mapping[str, ArrayLike]
    ) -> tuple[jax.Array, jax.Array]:
        """The polarizations of every source in one pass, each ``(N, F)``."""


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
    if not source_parameters:
        raise ValueError("sources must have at least one column")
    columns = {name: jnp.asarray(value) for name, value in source_parameters.items()}
    counts = {name: value.shape[:1] for name, value in columns.items()}
    if len(set(counts.values())) != 1 or any(len(c) != 1 for c in counts.values()):
        raise ValueError(f"sources must be one-dimensional of one length, got {counts}")
    (count,) = next(iter(counts.values()))
    if count == 0:
        # There is no chunk to map, and lax.map needs at least one.
        return generate(columns)
    full = count - count % chunk_size
    pieces = []
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
