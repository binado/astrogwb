from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "FrequencySpacing",
    "apply_frequency_mask",
    "bin_widths",
    "broadcast_along_axis",
    "frequency_grid",
    "frequency_mask",
    "log_frequency_grid",
    "noise_weighted_inner_product",
    "uniform_frequency_grid",
    "validate_frequency_grid",
]

#: How a frequency grid is laid out; the values of
#: ``WaveformMetadata.frequency_spacing``.
FrequencySpacing = Literal["linear", "log"]

GRID_SPACING_TOLERANCE_ULP = 64.0


def uniform_frequency_grid(
    minimum_frequency: float, maximum_frequency: float, df: float
) -> np.ndarray:
    """Return the inclusive uniform grid from ``minimum_frequency`` by ``df``.

    Parameters
    ----------
    minimum_frequency
        Lower frequency bound in Hz.
    maximum_frequency
        Upper frequency bound in Hz.
    df
        Frequency resolution in Hz.
    """
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


def _check_log_grid_arguments(minimum: float, maximum: float, df: float) -> None:
    if not (np.isfinite(minimum) and np.isfinite(maximum)):
        raise ValueError("frequency bounds must be finite")
    if minimum <= 0.0:
        raise ValueError("a logarithmic grid needs minimum_frequency > 0")
    if maximum <= minimum:
        raise ValueError("maximum_frequency must be greater than minimum_frequency")
    if not np.isfinite(df) or df <= 0.0:
        raise ValueError("df must be a finite positive scalar")


def _geometric_tail(start: float, maximum: float, df: float) -> np.ndarray:
    """Return the geometric points above ``start``, ending exactly at ``maximum``.

    The step count is the smallest that keeps the first geometric gap,
    ``start * (ratio - 1)``, no larger than ``df``; the ratio is then set so the
    last point lands on ``maximum``.
    """
    log_step = np.log1p(df / start)
    num_steps = max(int(np.ceil(np.log(maximum / start) / log_step)), 1)
    ratio = (maximum / start) ** (1.0 / num_steps)
    tail = start * ratio ** np.arange(1, num_steps + 1, dtype=np.float64)
    tail[-1] = maximum
    return tail


def log_frequency_grid(
    minimum_frequency: float, maximum_frequency: float, df: float
) -> np.ndarray:
    """Return the purely geometric grid from ``minimum_frequency`` to the maximum.

    ``df`` is the bin width at ``minimum_frequency``, so the log step is
    ``df / minimum_frequency``; the last point is exactly ``maximum_frequency``.

    Parameters
    ----------
    minimum_frequency
        Lower frequency bound in Hz.
    maximum_frequency
        Upper frequency bound in Hz.
    df
        Frequency resolution in Hz.
    """
    minimum = float(minimum_frequency)
    maximum = float(maximum_frequency)
    spacing = float(df)
    _check_log_grid_arguments(minimum, maximum, spacing)
    return np.concatenate([[minimum], _geometric_tail(minimum, maximum, spacing)])


def frequency_grid(
    spacing: FrequencySpacing,
    minimum_frequency: float,
    maximum_frequency: float,
    df: float,
) -> np.ndarray:
    """Return the grid ``spacing`` names, validated as a frequency grid.

    Parameters
    ----------
    spacing
        Frequency grid spacing mode.
    minimum_frequency
        Lower frequency bound in Hz.
    maximum_frequency
        Upper frequency bound in Hz.
    df
        Frequency resolution in Hz.
    """
    grid_builder = uniform_frequency_grid if spacing == "linear" else log_frequency_grid
    return validate_frequency_grid(
        grid_builder(minimum_frequency, maximum_frequency, df)
    )


def validate_frequency_grid(frequencies: ArrayLike) -> np.ndarray:
    """Return ``frequencies`` as a float64 array after checking it is a grid.

    A grid is one-dimensional, finite and strictly increasing; the spacing may
    vary.

    Parameters
    ----------
    frequencies
        Frequency grid in Hz.
    """
    array = np.asarray(frequencies, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(
            f"frequencies must be one-dimensional; received shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("frequencies must be finite")
    if array.size >= 2 and np.any(np.diff(array) <= 0.0):
        raise ValueError("frequencies must be strictly increasing")
    return array


def bin_widths(frequencies: ArrayLike) -> jax.Array:
    r"""Width :math:`\Delta f_i` of each bin of a frequency grid.

    Bin edges sit halfway between neighbouring frequencies, and the two end
    bins are as wide as their one neighbouring gap:

    .. math::

        \Delta f_i = \tfrac12 (f_{i+1} - f_{i-1}), \qquad
        \Delta f_0 = f_1 - f_0, \qquad
        \Delta f_{F-1} = f_{F-1} - f_{F-2}.

    Widths belong to the whole grid: compute them on the full axis and mask
    afterwards, or bins at the edge of a gap get the wrong width. Values are
    trusted (traceable); only the shape is checked.

    Parameters
    ----------
    frequencies
        Frequency grid in Hz.
    """
    grid = jnp.asarray(frequencies)
    if grid.ndim != 1:
        raise ValueError(
            f"frequencies must be one-dimensional; received shape {grid.shape}"
        )
    if grid.shape[0] < 2:
        raise ValueError("a frequency grid needs at least two bins to have a width")
    gaps = jnp.diff(grid)
    return jnp.concatenate([gaps[:1], (gaps[:-1] + gaps[1:]) / 2.0, gaps[-1:]])


def broadcast_along_axis(values: jax.Array, ndim: int, axis: int) -> jax.Array:
    """Reshape a per-frequency vector so it lines up with ``axis`` of an ``ndim`` array.

    Parameters
    ----------
    values
        Input values.
    ndim
        Number of dimensions in the target array.
    axis
        Frequency-axis index.
    """
    shape = [1] * ndim
    shape[axis] = values.shape[0]
    return jnp.reshape(values, shape)


def frequency_mask(
    frequencies: jax.Array,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> jax.Array:
    """Boolean mask selecting the bins inside the inclusive band.

    Parameters
    ----------
    frequencies
        Frequency grid in Hz.
    fmin
        Lower frequency bound in Hz.
    fmax
        Upper frequency bound in Hz.
    """
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

    Compress only after grid-dependent quantities such as :func:`bin_widths`
    have been computed on the full axis.

    Parameters
    ----------
    mask
        Boolean mask of entries to include.
    axis
        Frequency-axis index.
    *arrays
        Arrays to mask.
    """
    return tuple(jnp.compress(mask, array, axis=axis) for array in arrays)


def noise_weighted_inner_product(
    a: jax.Array,
    b: jax.Array,
    psd: jax.Array,
    frequencies: ArrayLike,
    *,
    axis: int = -1,
) -> jax.Array:
    r"""Discrete noise-weighted inner product for a diagonal Gaussian noise model.

    .. math::

        (a|b) = \sum_i \Delta f_i \frac{a_i^* b_i}{S_i^2},

    with the bin widths :math:`\Delta f_i` of the frequency grid. The
    observation time enters separately, in
    :func:`astrogwb.background.spectral_snr_squared`.

    Parameters
    ----------
    a, b
        Arrays whose ``axis`` runs over frequency bins; ``a`` is conjugated.
    psd
        Power spectral density :math:`S_i`, broadcastable against ``a`` and
        ``b``.
    frequencies
        The grid ``axis`` runs over, in Hz, one entry per bin. Widths come from
        :func:`bin_widths`.
    axis
        Axis to contract over. Defaults to the trailing axis, so leading batch
        dimensions broadcast.
    """
    terms = jnp.conj(a) * b / psd**2
    widths = broadcast_along_axis(bin_widths(frequencies), terms.ndim, axis)
    return jnp.sum(widths * terms, axis=axis)
