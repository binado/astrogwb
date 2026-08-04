from __future__ import annotations

import jax
import jax.numpy as jnp


def inner_product(
    a: jax.Array,
    b: jax.Array,
    effective_psd: jax.Array,
    observation_time_sec: float | jax.Array,
    df: float | jax.Array,
) -> jax.Array:
    r"""Discrete frequency-domain inner product for a diagonal Gaussian noise model.

    :math:`\langle a, b \rangle = 2 T \Delta f \sum_i a_i b_i / \mathrm{effective\_psd}_i^2`,

    where :math:`\sigma_i = \mathrm{effective\_psd}_i / \sqrt{2 T \Delta f}` and
    observation time :math:`T` is in seconds with bin width :math:`\Delta f =`
    ``df`` in Hz.

    With :math:`a = b` equal to a strain spectral density :math:`S_h`,
    :math:`\langle S_h, S_h \rangle` is matched-filter
    :math:`\mathrm{SNR}^2`; see :func:`spectral_snr_squared`.
    """
    prefactor = 2.0 * observation_time_sec * df
    return prefactor * jnp.sum(a * b / effective_psd**2)


def spectral_snr_squared(
    spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time_sec: float | jax.Array,
    df: float | jax.Array,
) -> jax.Array:
    r"""Discrete matched-filter :math:`\mathrm{SNR}^2` for a diagonal Gaussian noise model.

    :math:`\mathrm{SNR}^2 = \langle S_h, S_h \rangle = \sum_i S_{h,i}^2 / \sigma_i^2`,

    where :math:`\sigma_i = \mathrm{effective\_psd}_i / \sqrt{2 T \Delta f}` with
    observation time :math:`T` in seconds and bin width :math:`\Delta f =` ``df`` in Hz.

    The per-bin :math:`\sigma` matches :func:`astrogwb.detector.gaussian_bin_scale` when
    ``df`` is the same width used there and ``observation_time_sec`` is the
    corresponding value in seconds.
    """
    return inner_product(
        spectral_density,
        spectral_density,
        effective_psd,
        observation_time_sec,
        df,
    )


def spectral_snr(
    spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time_sec: float | jax.Array,
    df: float | jax.Array,
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
        )
    )
