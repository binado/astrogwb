"""Fixed-size source chunks: the one shape every stage compiles for.

A Poisson count differs per realization, and every distinct array shape is a
compile. So each stage works through its sources in pieces of one static
``chunk_size``, the last piece padded: a realization of any count reuses the
same compiled body, and ``chunk_size`` changes cost, never the result.

Padding rows repeat the chunk's first source -- a physical source, so a waveform
stays finite -- and the polarizations of padding rows are zero, so they add
nothing to any cross-power.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from astrogwb.populations import Population, joint_model
from astrogwb.populations.evaluation import evaluate_sources
from astrogwb.utils.numpyro import sample_model

__all__ = [
    "iter_rows",
    "pad_rows",
    "sample_extrinsic_by_key",
    "sample_sources_by_key",
]


def pad_rows(
    columns: Mapping[str, NDArray[Any]], start: int, size: int
) -> tuple[dict[str, NDArray[Any]], int]:
    """Rows ``start`` to ``start + size`` of ``columns``, padded, and how many are real.

    Padding rows repeat row ``start``.
    """
    rows = {
        name: np.asarray(values)[start : start + size]
        for name, values in columns.items()
    }
    valid = next(iter(rows.values())).shape[0]
    if valid < size:
        rows = {
            name: np.concatenate([values, np.repeat(values[:1], size - valid, axis=0)])
            for name, values in rows.items()
        }
    return rows, valid


def sample_sources_by_key(
    population: Population,
    params: Mapping[str, Any],
    key: jax.Array,
    start: jax.Array,
    size: int,
) -> dict[str, jax.Array]:
    """Events ``start`` to ``start + size``, event ``i`` drawn from ``fold_in(key, i)``.

    The counterpart of :func:`~astrogwb.populations.evaluation.sample_sources`
    with the same replay -- sampled sites are drawn, then every returned column,
    deterministics included, is recomputed through
    :func:`~astrogwb.populations.evaluation.evaluate_sources` -- but with one key
    per event instead of ``split(key, size)``. Splitting a key into ``n`` and
    into ``m > n`` children does not share a prefix, so there the drawn events
    would depend on ``size``; here an event depends on its index alone, so a
    population can be drawn in pieces of any size. ``start`` is a ``uint32``
    scalar, traced, so every piece of one ``size`` shares a compile.
    """
    event_keys = _event_keys(key, start, size)
    # Built once, outside the vmap: the hyperparameters are unbatched, so the
    # redshift grid is computed once, not once per event.
    model = joint_model(*population(params))
    sampled = jax.vmap(lambda k: sample_model(model, k))(event_keys)
    _, outputs = evaluate_sources(model, sampled, density_sites=())
    return outputs


def _event_keys(key: jax.Array, start: jax.Array, size: int) -> jax.Array:
    """``fold_in(key, i)`` for ``i`` in ``start .. start + size``."""
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(
        start + jnp.arange(size, dtype=jnp.uint32)
    )


def sample_extrinsic_by_key(
    key: jax.Array, start: jax.Array, size: int
) -> dict[str, jax.Array]:
    r"""Isotropic extrinsic angles of events ``start`` to ``start + size``.

    Event ``i`` is drawn from ``fold_in(key, i)``, as in
    :func:`sample_sources_by_key`: ``declination`` as arcsin of a uniform draw on :math:`[-1, 1]`
    and ``right_ascension`` on :math:`[0, 2\pi)` (an isotropic celestial direction),
    ``polarization_angle`` on :math:`[0, \pi)` and ``coa_phase`` on :math:`[0, 2\pi)`.
    """

    def one(event_key: jax.Array) -> dict[str, jax.Array]:
        keys = jax.random.split(event_key, 4)
        return {
            "declination": jnp.arcsin(
                jax.random.uniform(keys[0], minval=-1.0, maxval=1.0)
            ),
            "right_ascension": jax.random.uniform(keys[1], maxval=2.0 * jnp.pi),
            "polarization_angle": jax.random.uniform(keys[2], maxval=jnp.pi),
            "coa_phase": jax.random.uniform(keys[3], maxval=2.0 * jnp.pi),
        }

    return jax.vmap(one)(_event_keys(key, start, size))


def iter_rows(count: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """``(start, stop)`` of every chunk of ``count`` rows; none for no rows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, count, chunk_size):
        yield start, min(start + chunk_size, count)
