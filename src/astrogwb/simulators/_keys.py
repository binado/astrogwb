"""The single place a seed becomes a JAX key."""

from __future__ import annotations

import jax
import numpy as np

__all__ = ["seed_key"]


def seed_key(seed: int | np.integer) -> jax.Array:
    """``jax.random.key`` of a 64-bit ``seed``.

    Seeds are ``uint64`` (see :func:`~astrogwb.simulators.core.split_seed`), so
    this needs ``jax_enable_x64``: without it JAX would silently truncate the
    seed to 32 bits, and two different seeds could draw the same realization.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "seed_key needs jax_enable_x64; JAX would truncate a 64-bit seed"
        )
    return jax.random.key(np.uint64(seed))
