from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "apply_frequency_mask",
    "frequency_mask",
    "noise_weighted_inner_product",
    "uniform_frequency_grid",
    "uniform_grid_spacing",
]

GRID_SPACING_TOLERANCE_ULP = 64.0


def uniform_frequency_grid(
    minimum_frequency: float, maximum_frequency: float, df: float
) -> np.ndarray:
    """Return the inclusive uniform grid from ``minimum_frequency`` by ``df``."""
    minimum = float(minimum_frequency)
    maximum = float(maximum_frequency)
    spacing = float(df)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("frequency bounds must be finite")
    if maximum < minimum:
        raise ValueError(
            "maximum_frequency must be greater than or equal to minimum_frequency"
        )
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("df must be a finite positive scalar")
    span_in_bins = (maximum - minimum) / spacing
    num_bins = int(np.floor(span_in_bins)) + 1
    next_frequency = minimum + spacing * num_bins
    tolerance = (
        GRID_SPACING_TOLERANCE_ULP
        * np.finfo(np.float64).eps
        * max(1.0, abs(minimum), abs(maximum), abs(next_frequency))
    )
    if next_frequency <= maximum + tolerance:
        num_bins += 1
    frequencies = minimum + spacing * np.arange(num_bins, dtype=np.float64)
    if np.isclose(frequencies[-1], maximum, rtol=0.0, atol=tolerance):
        frequencies[-1] = maximum
    return frequencies


def uniform_grid_spacing(frequencies: ArrayLike) -> float:
    """Measure the bin width of a uniform frequency grid.

    The inverse of :func:`uniform_frequency_grid`: that function turns
    ``(f_min, f_max, df)`` into a grid, this one reads ``df`` back off it.
    Deriving rather than storing is what keeps a catalog's bin width from
    drifting away from the axis a waveform backend actually produced.

    Returns ``frequencies[1] - frequencies[0]``, not the mean spacing
    ``(frequencies[-1] - frequencies[0]) / (n - 1)`` -- the former is exact
    for Ripple's ``arange(n) * delta_f`` grid and reproduces the backend's
    ``delta_f`` bit for bit, while the latter would make the result depend on
    ``n``.

    Never call this on a *masked* analysis frequency array -- see
    :func:`apply_frequency_mask`.
    """
    array = np.asarray(frequencies, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(
            f"frequencies must be one-dimensional; received shape {array.shape}"
        )
    if array.size < 2:
        raise ValueError("a frequency grid needs at least two bins to have a spacing")
    gaps = np.diff(array)
    if np.any(gaps <= 0.0):
        raise ValueError("frequencies must be strictly increasing")
    spacing = float(gaps[0])
    # Absolute and scaled to the frequencies, not to the spacing: an analytic
    # `f_min + df * arange(n)` grid carries ~1 ulp of the *largest* frequency
    # in each gap (Ripple's `arange(n) * delta_f` is bit-identical, so this
    # tolerance only ever has to cover the analytic construction). Scaling to
    # `df` itself would be orders of magnitude too tight and would reject the
    # package's own analytic catalogs.
    tolerance = (
        GRID_SPACING_TOLERANCE_ULP
        * np.finfo(np.float64).eps
        * max(1.0, abs(float(array[0])), abs(float(array[-1])))
    )
    if not np.allclose(gaps, spacing, rtol=0.0, atol=tolerance):
        raise ValueError("frequencies are not uniform")
    return spacing


def frequency_mask(
    frequencies: jax.Array,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> jax.Array:
    """Boolean mask selecting the bins inside the inclusive band."""
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

    The mask need not select a contiguous run: every surviving bin keeps its
    own width, which is :func:`uniform_grid_spacing` measured at the
    *catalog's* grid, before masking. Nothing downstream measures the spacing
    of the masked grid -- ``df`` is always passed explicitly, never derived
    from the analysis band. Calling :func:`uniform_grid_spacing` on a masked
    grid is a bug: a non-contiguous selection is not uniform and would raise,
    while a contiguous one would silently return the right number for the
    wrong reason.
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
