"""Deterministic keys for simulation realizations."""

import jax
import numpy as np


def _require_x64() -> None:

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "realization keys need jax_enable_x64; JAX would truncate a 64-bit seed"
        )


def realization_key(seed: int | np.integer, index: int) -> jax.Array:
    """Derive the key for a realization by folding its index into the seed."""

    if index < 0:
        raise ValueError("index must be non-negative")
    _require_x64()
    return jax.random.fold_in(jax.random.key(np.uint64(seed)), np.uint32(index))
