from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def polarization_power(plus: ArrayLike, cross: ArrayLike) -> jax.Array:
    """Reduce ``(n, F)`` complex polarization arrays to ``(F, n)`` float64 power.

    Returns ``|h+|^2 + |hx|^2`` per sample and frequency, transposed to the
    on-disk ``(frequency, sample)`` layout.
    """
    power = jnp.abs(jnp.asarray(plus)) ** 2 + jnp.abs(jnp.asarray(cross)) ** 2
    return power.T.astype(jnp.float64)
