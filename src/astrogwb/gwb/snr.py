from __future__ import annotations

import jax
import jax.numpy as jnp


def spectral_snr_squared_per_bin(
    spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time_sec: float | jax.Array,
    df: float | jax.Array,
) -> jax.Array:
    r"""Per-bin contribution to :func:`spectral_snr_squared`.

    .. math::

        \Delta\mathrm{SNR}^2_i = 2 T \Delta f \frac{S_{h,i}^2}{S_{\mathrm{eff},i}^2}

    Summing along the frequency axis recovers :math:`\mathrm{SNR}^2`. The
    arrays are not contracted here, so leading batch dimensions broadcast.
    """
    return 2.0 * observation_time_sec * df * spectral_density**2 / effective_psd**2


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
        ),
        axis=axis,
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
