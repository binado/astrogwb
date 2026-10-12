from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from astrogwb.cosmology import hubble_constant_si
from astrogwb.detector import log_frequency_noise_scale


def spectral_density(
    polarization_power: jax.Array,
    weights: jax.Array,
    total_merger_rate: float | jax.Array,
) -> jax.Array:
    return total_merger_rate * jnp.dot(polarization_power, weights) / weights.shape[0]


def omega_gw_from_spectral_density(
    spectral_density: jax.Array,
    frequencies: jax.Array,
    *,
    hubble_constant: float,
) -> jax.Array:
    coefficient = 4.0 * jnp.pi**2 / (3.0 * hubble_constant_si(hubble_constant) ** 2)
    return coefficient * frequencies**3 * spectral_density


def spectral_density_from_omega_gw(
    omega_gw: jax.Array,
    frequencies: jax.Array,
    *,
    hubble_constant: float,
) -> jax.Array:
    coefficient = 3.0 * hubble_constant_si(hubble_constant) ** 2 / (4.0 * jnp.pi**2)
    return coefficient * omega_gw / frequencies**3


def characteristic_strain(
    spectral_density: ArrayLike,
    frequencies: ArrayLike,
) -> jax.Array:
    r"""Dimensionless characteristic strain of the background.

    .. math::

        h_c(f) = \sqrt{2 f\, S_h(f)}

    This is Eq. 47 of Belgacem et al. (arXiv:2411.04029). Against
    :func:`characteristic_noise`,
    :math:`(h_c / h_n)^4 = d\mathrm{SNR}^2 / d\ln f`
    (:func:`spectral_snr_squared_per_log_frequency`).

    Parameters
    ----------
    spectral_density
        Strain spectral density :math:`S_h` in Hz^-1.
    frequencies
        Frequencies in Hz, broadcasting against ``spectral_density``.

    Returns
    -------
    jax.Array
        :math:`h_c` with the broadcast shape.
    """
    return jnp.sqrt(2.0 * jnp.asarray(frequencies) * jnp.asarray(spectral_density))


def characteristic_noise(
    effective_psd: ArrayLike,
    frequencies: ArrayLike,
    observation_time: float,
) -> jax.Array:
    r"""Characteristic noise strain of a stochastic background search.

    .. math::

        h_n(f) = \sqrt{2 f\, \sigma_{\ln f}(f)}
               = \frac{\sqrt{2 f\, S_{\mathrm{eff}}(f)}}{(2 T f)^{1/4}}

    This is Eq. 48 of Belgacem et al. (arXiv:2411.04029), with
    :math:`\sigma_{\ln f}` from :func:`log_frequency_noise_scale`. A spectrum
    with :math:`h_c = h_n` accrues unit :math:`d\mathrm{SNR}^2/d\ln f`, and
    :math:`\mathrm{SNR}^2 = \int d\ln f\, (h_c/h_n)^4` (Eq. 49).

    Parameters
    ----------
    effective_psd
        Network effective PSD in Hz^-1, aligned with ``frequencies``.
    frequencies
        Frequencies in Hz.
    observation_time
        Observation time in years. Converted to seconds internally.

    Returns
    -------
    jax.Array
        :math:`h_n`, aligned with ``frequencies``.
    """
    frequencies = jnp.asarray(frequencies)
    scale = log_frequency_noise_scale(
        jnp.asarray(effective_psd), frequencies, observation_time
    )
    return jnp.sqrt(2.0 * frequencies * scale)
