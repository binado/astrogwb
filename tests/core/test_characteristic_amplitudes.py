from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.gwb import (
    characteristic_noise,
    characteristic_strain,
    spectral_snr_squared,
    spectral_snr_squared_per_log_frequency,
)
from astrogwb.utils import years_to_seconds

GRIDS = {
    "decade": jnp.geomspace(1.0, 10.0, 7),
    "audio-band": jnp.geomspace(2.0, 2048.0, 40),
    "uneven": jnp.array([3.0, 4.5, 11.0, 30.0, 95.0, 400.0]),
}


def _power_law(frequencies: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    spectral_density = 1e-47 * (frequencies / 10.0) ** (2.0 / 3.0)
    effective_psd = 1e-45 * (1.0 + (20.0 / frequencies) ** 4)
    return spectral_density, effective_psd


@pytest.mark.parametrize("grid", GRIDS.values(), ids=GRIDS.keys())
@pytest.mark.parametrize("observation_time", [0.5, 3.0])
def test_amplitude_ratio_fourth_power_is_snr_density_per_log_frequency(
    grid: jnp.ndarray, observation_time: float
) -> None:
    spectral_density, noise = _power_law(grid)

    ratio = characteristic_strain(spectral_density, grid) / characteristic_noise(
        noise, grid, observation_time
    )
    density = spectral_snr_squared_per_log_frequency(
        spectral_density, noise, grid, years_to_seconds(observation_time)
    )

    np.testing.assert_allclose(np.asarray(ratio**4), np.asarray(density), rtol=1e-6)


@pytest.mark.parametrize("observation_time", [0.5, 3.0])
def test_log_integral_of_amplitude_ratio_reproduces_snr_squared(
    observation_time: float,
) -> None:
    grid = jnp.geomspace(2.0, 2048.0, 4096)
    spectral_density, noise = _power_law(grid)

    ratio = characteristic_strain(spectral_density, grid) / characteristic_noise(
        noise, grid, observation_time
    )
    integral = jnp.trapezoid(ratio**4, jnp.log(grid))
    expected = spectral_snr_squared(
        spectral_density, noise, years_to_seconds(observation_time), grid
    )

    np.testing.assert_allclose(float(integral), float(expected), rtol=1e-2)


def test_characteristic_amplitudes_broadcast_over_leading_draw_axis() -> None:
    grid = GRIDS["audio-band"]
    spectral_density, noise = _power_law(grid)
    draws = jnp.stack([spectral_density, 2.0 * spectral_density])

    strain = characteristic_strain(draws, grid)

    assert strain.shape == draws.shape
    assert characteristic_noise(noise, grid, 1.0).shape == grid.shape
