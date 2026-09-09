from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "apply_frequency_mask",
    "frequency_mask",
    "noise_weighted_inner_product",
    "ripple_frequency_grid",
    "uniform_frequency_grid",
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


def ripple_frequency_grid(
    *,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float,
    frequency_resolution: float,
) -> np.ndarray:
    """Reproduce Ripple's generated positive-frequency grid in a band."""
    sampling = float(sampling_frequency)
    resolution = float(frequency_resolution)
    minimum = float(minimum_frequency)
    maximum = float(maximum_frequency)
    if not np.isfinite(sampling) or sampling <= 0.0:
        raise ValueError("sampling_frequency must be finite and positive")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("frequency_resolution must be a finite positive scalar")
    segment_duration = float(2.0 ** np.ceil(np.log2(1.0 / resolution)))
    n_samples = round(segment_duration * sampling)
    if n_samples <= 0:
        raise ValueError("sampling_frequency produces no Ripple samples")
    effective_df = sampling / n_samples
    frequencies = effective_df * np.arange(n_samples // 2 + 1, dtype=np.float64)
    return frequencies[(frequencies >= minimum) & (frequencies <= maximum)]


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
    own width, which is the catalog's ``df`` attribute. Nothing downstream
    measures the spacing of the masked grid -- ``df`` is always passed
    explicitly, never derived from the analysis band.
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
