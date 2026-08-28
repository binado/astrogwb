from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from array_api_compat import array_namespace

if TYPE_CHECKING:
    import jax
    from numpy.typing import NDArray

__all__ = [
    "apply_frequency_mask",
    "frequency_mask",
    "frequency_spacing",
    "noise_weighted_inner_product",
]


@overload
def frequency_spacing(frequencies: jax.Array) -> jax.Array: ...


@overload
def frequency_spacing(frequencies: NDArray[Any]) -> NDArray[Any]: ...


def frequency_spacing(
    frequencies: jax.Array | NDArray[Any],
) -> jax.Array | NDArray[Any]:
    """Mean consecutive spacing Δf from a frequency grid (Hz)."""
    xp = array_namespace(frequencies)
    return xp.mean(xp.diff(frequencies))


@overload
def frequency_mask(
    frequencies: jax.Array,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> jax.Array: ...


@overload
def frequency_mask(
    frequencies: NDArray[Any],
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> NDArray[Any]: ...


def frequency_mask(
    frequencies: jax.Array | NDArray[Any],
    *,
    fmin: float | None = None,
    fmax: float | None = None,
) -> jax.Array | NDArray[Any]:
    xp = array_namespace(frequencies)
    mask = xp.ones_like(frequencies, dtype=xp.bool)
    if fmin is not None:
        mask = mask & (frequencies >= fmin)
    if fmax is not None:
        mask = mask & (frequencies <= fmax)
    return mask


@overload
def apply_frequency_mask(
    mask: jax.Array,
    *arrays: jax.Array,
    axis: int = 0,
) -> tuple[jax.Array, ...]: ...


@overload
def apply_frequency_mask(
    mask: NDArray[Any],
    *arrays: NDArray[Any],
    axis: int = 0,
) -> tuple[NDArray[Any], ...]: ...


def apply_frequency_mask(
    mask: jax.Array | NDArray[Any],
    *arrays: jax.Array | NDArray[Any],
    axis: int = 0,
) -> tuple[jax.Array | NDArray[Any], ...]:
    """Apply a boolean frequency mask along ``axis`` of every array.

    Defaults to ``axis=0`` to match this package's ``(F, ...)`` layout
    (e.g. polarization power of shape ``(F, N)``).

    Selection uses ``take`` with ``nonzero`` indices because ``compress``
    is not part of the array API standard; the result is identical for a
    boolean mask and keeps the helpers namespace-neutral.
    """
    xp = array_namespace(mask)
    indices = xp.nonzero(mask)[0]
    return tuple(xp.take(array, indices, axis=axis) for array in arrays)


@overload
def noise_weighted_inner_product(
    a: jax.Array,
    b: jax.Array | NDArray[Any],
    psd: jax.Array | NDArray[Any],
    df: float | jax.Array | NDArray[Any],
    *,
    axis: int = -1,
) -> jax.Array: ...


@overload
def noise_weighted_inner_product(
    a: NDArray[Any],
    b: jax.Array | NDArray[Any],
    psd: jax.Array | NDArray[Any],
    df: float | jax.Array | NDArray[Any],
    *,
    axis: int = -1,
) -> NDArray[Any]: ...


def noise_weighted_inner_product(
    a: jax.Array | NDArray[Any],
    b: jax.Array | NDArray[Any],
    psd: jax.Array | NDArray[Any],
    df: float | jax.Array | NDArray[Any],
    *,
    axis: int = -1,
) -> jax.Array | NDArray[Any]:
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
    xp = array_namespace(a)
    return df * xp.sum(a * b / psd**2, axis=axis)
