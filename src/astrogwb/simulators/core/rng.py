"""Batched JAX keys: the one way a seed becomes the keys a simulator is called on."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["batch_keys"]


def batch_keys(seed: int | np.integer, n: int) -> Any:
    """``n`` keys of ``seed``: ``fold_in(key(seed), arange(n))``.

    Key ``i`` depends on ``(seed, i)`` alone, so asking for more keys extends
    the batch without changing the ones already there (prefix-stable). This is
    how a stochastic simulator's batched call is seeded: ``simulator(keys)[i]``
    and ``simulator(keys[i : i + 1])`` see the same randomness.

    Seeds are 64-bit, so this needs ``jax_enable_x64``: without it JAX would
    silently truncate the seed to 32 bits, and two different seeds could share a
    realization.
    """
    import jax

    if n < 0:
        raise ValueError("n must be non-negative")
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "batch_keys needs jax_enable_x64; JAX would truncate a 64-bit seed"
        )
    base = jax.random.key(np.uint64(seed))
    return jax.vmap(lambda i: jax.random.fold_in(base, i))(
        jax.numpy.arange(n, dtype=jax.numpy.uint32)
    )
