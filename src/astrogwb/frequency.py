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
    "loglinear_frequency_grid",
    "noise_weighted_inner_product",
    "uniform_frequency_grid",
    "validate_frequency_grid",
]

#: How a frequency grid is laid out; the values of
#: ``WaveformMetadata.frequency_spacing``.
FrequencySpacing = Literal["linear", "log", "loglinear"]

GRID_SPACING_TOLERANCE_ULP = 64.0
#: Relative slack when counting how many linear steps fit below the turnover, so
#: a turnover that is a whole number of steps from the minimum is not lost to
#: rounding.
_TURNOVER_STEP_TOLERANCE = 1e-9


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
    num_steps = max(int(np.ceil(np.log(maximum / start) / np.log1p(df / start))), 1)
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
    """
    minimum = float(minimum_frequency)
    maximum = float(maximum_frequency)
    spacing = float(df)
    _check_log_grid_arguments(minimum, maximum, spacing)
    return np.concatenate([[minimum], _geometric_tail(minimum, maximum, spacing)])


def loglinear_frequency_grid(
    minimum_frequency: float,
    maximum_frequency: float,
    df: float,
    turnover_frequency: float,
) -> np.ndarray:
    """Return a grid linear up to the turnover and geometric above it.

    Below the turnover the grid is ``minimum_frequency + df * k``. The turn
    itself is the last of those points not above ``turnover_frequency`` (the
    *effective* turnover, at most one ``df`` below the request). Above it the
    grid is geometric, with a ratio chosen so the first geometric gap is at
    most ``df`` (continuous with the linear spacing at the turn) and the last
    point is exactly ``maximum_frequency``.
    """
    minimum = float(minimum_frequency)
    maximum = float(maximum_frequency)
    spacing = float(df)
    turnover = float(turnover_frequency)
    _check_log_grid_arguments(minimum, maximum, spacing)
    if not np.isfinite(turnover) or not minimum < turnover < maximum:
        raise ValueError(
            "turnover_frequency must lie strictly between minimum_frequency and "
            "maximum_frequency"
        )
    num_linear_steps = int(
        np.floor((turnover - minimum) / spacing + _TURNOVER_STEP_TOLERANCE)
    )
    linear = minimum + spacing * np.arange(num_linear_steps + 1, dtype=np.float64)
    return np.concatenate([linear, _geometric_tail(linear[-1], maximum, spacing)])


def frequency_grid(
    spacing: FrequencySpacing,
    minimum_frequency: float,
    maximum_frequency: float,
    df: float,
    turnover_frequency: float | None = None,
) -> np.ndarray:
    """Return the grid ``spacing`` names, validated as a frequency grid.

    The single entry point the waveform generators use, so a
    ``WaveformMetadata`` record maps to one grid. ``turnover_frequency`` is
    required for ``"loglinear"`` and must be ``None`` otherwise.
    """
    if spacing == "loglinear":
        if turnover_frequency is None:
            raise ValueError("the 'loglinear' spacing needs a turnover_frequency")
        grid = loglinear_frequency_grid(
            minimum_frequency, maximum_frequency, df, turnover_frequency
        )
    else:
        if turnover_frequency is not None:
            raise ValueError(
                f"turnover_frequency is only used by the 'loglinear' spacing, "
                f"not {spacing!r}"
            )
        grid_builder = (
            uniform_frequency_grid if spacing == "linear" else log_frequency_grid
        )
        grid = grid_builder(minimum_frequency, maximum_frequency, df)
    return validate_frequency_grid(grid)


def validate_frequency_grid(frequencies: ArrayLike) -> np.ndarray:
    """Return ``frequencies`` as a float64 array after checking it is a grid.

    A grid is one-dimensional, finite and strictly increasing. Nothing else is
    required: the spacing may vary from bin to bin, and every quantity that
    needs a bin width derives it from the grid with :func:`bin_widths`. A
    one-bin grid is valid; it simply has no width to report.
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

    This is the number of Fourier modes per bin divided by the observation
    time, so it is the width both the Gaussian noise scale and the Riemann sum
    of an SNR use. On a uniform grid every width equals the grid spacing
    exactly; on a non-uniform grid the sum over bins is a midpoint-rule
    quadrature, accurate to second order in the local spacing.

    Widths belong to the *grid*, not to a band inside it: derive them from the
    catalog's full frequency axis and mask afterwards. Selecting bins first and
    then calling this on the selection would give the bins at the edge of a gap
    the wrong width.

    The values are trusted: the grid is assumed strictly increasing, as
    :func:`validate_frequency_grid` checks where a grid enters the package, so
    this stays traceable under :func:`jax.jit`. Only the shape is checked,
    raising ``ValueError`` for a grid that is not one-dimensional or has fewer
    than two bins, which has no width.
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
    """Reshape a per-frequency vector so it lines up with ``axis`` of an ``ndim`` array."""
    shape = [1] * ndim
    shape[axis] = values.shape[0]
    return jnp.reshape(values, shape)


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

    The mask need not select a contiguous run. Compress arrays *after* any
    quantity that depends on the grid has been computed on the full axis: a bin
    width is :func:`bin_widths` of the catalog's whole grid, so calling it on a
    masked grid would give the bins at the edge of a gap the wrong width.
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

    the Riemann sum approximating :math:`\int \mathrm{d}f\, a(f)\, b(f) /
    S(f)^2` with the bin widths :math:`\Delta f_i` of the frequency grid.
    Frequencies are discrete throughout this package, so the sum is the
    definition and the integral is what it approximates, not the other way
    round.

    Depends only on the noise curve and the band. The observation time enters
    separately, in :func:`astrogwb.gwb.spectral_snr_squared`.

    Parameters
    ----------
    a, b:
        Arrays whose ``axis`` runs over frequency bins. ``a`` is conjugated, so
        the product is the Hermitian one and complex strain enters correctly;
        for real inputs this is a no-op.
    psd:
        Power spectral density :math:`S_i`, broadcastable against ``a`` and
        ``b``.
    frequencies:
        The grid ``axis`` runs over, in Hz, one entry per bin. Widths come from
        :func:`bin_widths`.
    axis:
        Axis to contract over. Defaults to the trailing axis, so leading batch
        dimensions broadcast.
    """
    terms = jnp.conj(a) * b / psd**2
    widths = broadcast_along_axis(bin_widths(frequencies), terms.ndim, axis)
    return jnp.sum(widths * terms, axis=axis)
