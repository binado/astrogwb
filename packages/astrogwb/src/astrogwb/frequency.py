from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "frequency_slice",
    "noise_weighted_inner_product",
]


def frequency_slice(
    frequencies: ArrayLike,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> slice:
    """Return the contiguous slice inside the inclusive frequency bounds."""
    values = np.asarray(frequencies)
    if values.ndim != 1:
        raise ValueError("frequencies must be one-dimensional")
    if values.size == 0:
        raise ValueError("frequencies must contain at least one bin")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError("frequencies must be strictly increasing")
    if fmin is not None and fmax is not None and fmin > fmax:
        raise ValueError(f"fmin ({fmin}) must be <= fmax ({fmax})")

    start = 0 if fmin is None else int(np.searchsorted(values, fmin, side="left"))
    stop = (
        values.shape[0]
        if fmax is None
        else int(np.searchsorted(values, fmax, side="right"))
    )
    if start >= stop:
        bounds = (
            f"[{fmin if fmin is not None else '-inf'}, "
            f"{fmax if fmax is not None else 'inf'}]"
        )
        raise ValueError(f"frequency band {bounds} contains no bins")
    return slice(start, stop)


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
