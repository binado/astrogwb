from __future__ import annotations

import jax
import jax.numpy as jnp


def _broadcast_scale_along_axis(
    scale: float | jax.Array,
    ndim: int,
    axis: int,
) -> jax.Array:
    """Expand a scalar or batch factor so it does not align to ``axis``.

    NumPy broadcasts trailing dimensions, so a ``(batch,)`` scale would
    otherwise multiply the frequency axis of a ``(batch, frequency)`` array.
    Inserting a length-1 axis at ``axis`` makes the factor apply per batch
    item, matching multiplication after a reduction along that axis.
    """
    array = jnp.asarray(scale)
    if array.ndim == 0 or array.ndim == ndim:
        return array
    axis_norm = axis if axis >= 0 else axis + ndim
    if array.ndim == ndim - 1:
        return jnp.expand_dims(array, axis_norm)
    raise ValueError(
        f"scale with shape {array.shape} cannot broadcast against "
        f"{ndim}-dimensional arrays along axis {axis}"
    )


def spectral_snr_squared_per_bin(
    spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time_sec: float | jax.Array,
    df: float | jax.Array,
    *,
    axis: int = -1,
) -> jax.Array:
    r"""Per-bin contribution to :func:`spectral_snr_squared`.

    .. math::

        \Delta\mathrm{SNR}^2_i = 2 T \Delta f \frac{S_{h,i}^2}{S_{\mathrm{eff},i}^2}

    Summing along ``axis`` recovers :math:`\mathrm{SNR}^2`. ``T`` and
    ``df`` are batch factors: a trailing ``(batch,)`` shape is expanded at
    ``axis`` rather than multiplied against the frequency bins.
    """
    ratio_squared = spectral_density**2 / effective_psd**2
    scale = _broadcast_scale_along_axis(
        2.0 * observation_time_sec * df,
        ratio_squared.ndim,
        axis,
    )
    return scale * ratio_squared


def spectral_snr_squared(
    spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time_sec: float | jax.Array,
    df: float | jax.Array,
    *,
    axis: int = -1,
) -> jax.Array:
    r"""Discrete matched-filter :math:`\mathrm{SNR}^2` for a diagonal Gaussian noise model.

    .. math::

        \mathrm{SNR}^2 = 2 T \, (S_h|S_h)
            = 2 T \Delta f \sum_i \frac{S_{h,i}^2}{S_{\mathrm{eff},i}^2}
            = \sum_i \frac{S_{h,i}^2}{\sigma_i^2}
            = \sum_i \Delta\mathrm{SNR}^2_i,

    with per-bin terms from :func:`spectral_snr_squared_per_bin`. The
    factor :math:`2T` is what makes the result dimensionless --
    :math:`(S_h|S_h)` carries units of Hz.

    The per-bin scale :math:`\sigma_i = S_{\mathrm{eff},i}/\sqrt{2 T \Delta f}`
    matches :func:`astrogwb.detector.gaussian_bin_scale` when ``df`` is the same
    width used there and ``observation_time_sec`` is the corresponding value in
    seconds -- note that function takes **years**.
    """
    return jnp.sum(
        spectral_snr_squared_per_bin(
            spectral_density,
            effective_psd,
            observation_time_sec,
            df,
            axis=axis,
        ),
        axis=axis,
    )


def spectral_snr_squared_per_log_frequency(
    spectral_density: jax.Array,
    effective_psd: jax.Array,
    frequencies: jax.Array,
    observation_time_sec: float | jax.Array,
) -> jax.Array:
    r"""SNR density per e-fold of frequency, independent of the grid.

    .. math::

        \frac{d\mathrm{SNR}^2}{d\ln f}
            = 2 T f \frac{S_h^2}{S_{\mathrm{eff}}^2}
            = \left(\frac{S_h}{\sigma_{\ln f}}\right)^2
            = \frac{f_i}{\Delta f}\,\Delta\mathrm{SNR}^2_i,

    with :math:`\sigma_{\ln f}` from
    :func:`astrogwb.detector.log_frequency_noise_scale` and
    :math:`\Delta\mathrm{SNR}^2_i` from :func:`spectral_snr_squared_per_bin`.
    Plotted against :math:`f` with a log x-axis and a linear y-axis, the area
    under the curve over a band is that band's share of :math:`\mathrm{SNR}^2`.
    ``frequencies`` broadcasts against the trailing (frequency) axis.
    """
    ratio_squared = spectral_density**2 / effective_psd**2
    return 2.0 * observation_time_sec * frequencies * ratio_squared


def spectral_snr(
    spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time_sec: float | jax.Array,
    df: float | jax.Array,
    *,
    axis: int = -1,
) -> jax.Array:
    r""":math:`\mathrm{SNR} = \sqrt{\mathrm{SNR}^2}` with :math:`\mathrm{SNR}^2` from
    :func:`spectral_snr_squared`.
    """
    return jnp.sqrt(
        spectral_snr_squared(
            spectral_density,
            effective_psd,
            observation_time_sec,
            df,
            axis=axis,
        )
    )
