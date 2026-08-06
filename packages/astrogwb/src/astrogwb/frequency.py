from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "apply_frequency_mask",
    "frequency_mask",
    "noise_weighted_inner_product",
]


def frequency_mask(
    frequencies: jax.Array,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> jax.Array:
    mask = jnp.ones_like(frequencies, dtype=bool)
    if fmin is not None:
        mask = mask & (frequencies >= fmin)
    if fmax is not None:
        mask = mask & (frequencies <= fmax)
    return mask


def apply_frequency_mask(
    mask: jax.Array,
    *arrays: jax.Array,
    axis: int = 0,
) -> tuple[jax.Array, ...]:
    """Apply a boolean frequency mask along ``axis`` of every array.

    Defaults to ``axis=0`` to match this package's ``(F, ...)`` layout
    (e.g. polarization power of shape ``(F, N)``).
    """
    return tuple(jnp.compress(mask, array, axis=axis) for array in arrays)


def noise_weighted_inner_product(
    a: jax.Array,
    b: jax.Array,
    psd: jax.Array,
    df: float | jax.Array,
    *,
    axis: int = -1,
) -> jax.Array:
    r"""Discrete noise-weighted inner product for a diagonal Gaussian noise model.

    .. math::

        (a|b) = \Delta f \sum_i \frac{a_i b_i}{S_i^2},

    the Riemann sum approximating :math:`\int \mathrm{d}f\, a(f)\, b(f) /
    S(f)^2` on a grid of spacing :math:`\Delta f`. Frequencies are
    discrete throughout this package, so the sum is the definition and the
    integral is what it approximates, not the other way round.

    Depends only on the noise curve and the band. The observation time enters
    separately, in :func:`astrogwb.gwb.spectral_snr_squared`.

    Parameters
    ----------
    a, b:
        Arrays whose ``axis`` runs over frequency bins.
    psd:
        Power spectral density :math:`S_i`, broadcastable against ``a`` and
        ``b``.
    df:
        Frequency bin width :math:`\Delta f` in Hz.
    axis:
        Axis to contract over. Defaults to the trailing axis, so leading batch
        dimensions broadcast.
    """
    return df * jnp.sum(a * b / psd**2, axis=axis)
