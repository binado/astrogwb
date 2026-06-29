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
    """Discrete frequency-domain inner product for a diagonal Gaussian noise model.

    ``⟨a, b⟩ = 2 T Δf Σ_i a_i b_i / effective_psd_i²``,

    where ``σ_i = effective_psd_i / √(2 T Δf)`` and observation time ``T`` is in
    seconds with bin width ``Δf =`` ``df`` in Hz.

    With ``a = b`` equal to a strain spectral density ``S_h``,
    ``⟨S_h, S_h⟩`` is matched-filter **SNR²**; see :func:`spectral_snr_squared`.
    """
    prefactor = 2.0 * observation_time_sec * df
    return prefactor * jnp.sum(a * b / effective_psd**2)


def spectral_snr_squared(
    spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time_sec: float | jax.Array,
    df: float | jax.Array,
) -> jax.Array:
    """Discrete matched-filter **SNR²** for a diagonal Gaussian noise model.

    ``SNR² = ⟨S_h, S_h⟩ = Σ_i S_{h,i}² / σ_i²``,

    where ``σ_i = effective_psd_i / √(2 T Δf)`` with observation time ``T`` in
    seconds and bin width ``Δf =`` ``df`` in Hz.

    The per-bin ``σ`` matches :func:`astrogwb.gwb.gaussian_bin_scale` when
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
    """``SNR = √(SNR²)`` with ``SNR²`` from :func:`spectral_snr_squared`."""
    return jnp.sqrt(
        spectral_snr_squared(
            spectral_density,
            effective_psd,
            observation_time_sec,
            df,
        )
    )
