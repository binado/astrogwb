from __future__ import annotations

import jax
import jax.numpy as jnp

from astrogwb.frequency import noise_weighted_inner_product


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
            = \sum_i \frac{S_{h,i}^2}{\sigma_i^2},

    with :math:`(\cdot|\cdot)` from :func:`astrogwb.frequency.noise_weighted_inner_product`. The
    factor :math:`2T` is what makes the result dimensionless --
    :math:`(S_h|S_h)` carries units of Hz.

    The per-bin scale :math:`\sigma_i = S_{\mathrm{eff},i}/\sqrt{2 T \Delta f}`
    matches :func:`astrogwb.detector.gaussian_bin_scale` when ``df`` is the same
    width used there and ``observation_time_sec`` is the corresponding value in
    seconds -- note that function takes **years**.
    """
    return (
        2.0
        * observation_time_sec
        * noise_weighted_inner_product(
            spectral_density, spectral_density, effective_psd, df, axis=axis
        )
    )


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
