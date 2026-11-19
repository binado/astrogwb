"""The optimal cross-correlation estimator against ``pygwb.postprocessing``.

pygwb works per baseline in energy-density units, dividing by the overlap
reduction function; the library keeps cross powers of mean
:math:`\\gamma_{ab} S_h`. The baseline comparison multiplies pygwb's sigma by
pygwb's own ORF, so it is independent of the ORF comparison; the network
comparisons use each side's ORF and inherit that residual.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PYGWB_POWER_FRACTION
from numpy.typing import NDArray
from tolerances import assert_close

from astrogwb.background.spectral import (
    omega_gw_from_spectral_density,
    spectral_density_from_omega_gw,
)
from astrogwb.detector import (
    Detector,
    DetectorNetworkConfig,
    effective_psd,
    gaussian_bin_scale,
    pair_bin_scale,
)
from astrogwb.statistics import cross_correlation_estimator


@pytest.mark.parametrize("pair", range(3), ids=["A-B", "A-C", "B-C"])
def test_pair_noise_scale_matches_pygwb_sigma(
    pair: int,
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
    hubble_constant: float,
    observation_years: float,
    exact_detectors: tuple[Detector, ...],
) -> None:
    cross_power_sigma = spectral_density_from_omega_gw(
        jnp.asarray(
            pygwb_reference["sigma"][pair] * np.abs(pygwb_reference["orf_exact"][pair])
        ),
        jnp.asarray(frequencies),
        hubble_constant=hubble_constant,
    )

    scale = pair_bin_scale(frequencies, exact_detectors, observation_years)

    assert_close(scale[pair], cross_power_sigma)


def test_network_noise_scale_matches_pygwb_combined_sigma(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
    hubble_constant: float,
    observation_years: float,
    exact_detectors: tuple[Detector, ...],
) -> None:
    combined_sigma = spectral_density_from_omega_gw(
        jnp.asarray(pygwb_reference["combined_sigma"]),
        jnp.asarray(frequencies),
        hubble_constant=hubble_constant,
    )

    scale = gaussian_bin_scale(
        jnp.asarray(effective_psd(frequencies, exact_detectors)),
        observation_years,
        frequencies,
    )

    assert_close(scale, combined_sigma)


def test_network_spectrum_matches_pygwb_combined_point_estimate(
    pygwb_reference: Mapping[str, NDArray[np.float64]],
    frequencies: NDArray[np.float64],
    hubble_constant: float,
    exact_network: DetectorNetworkConfig,
) -> None:
    pair_spectra = pygwb_reference["csd"] / PYGWB_POWER_FRACTION

    omega = omega_gw_from_spectral_density(
        jnp.asarray(
            cross_correlation_estimator(
                pair_spectra, frequencies, exact_network.build_detectors()
            )
        ),
        jnp.asarray(frequencies),
        hubble_constant=hubble_constant,
    )

    # In units of the estimator's own standard deviation: the estimate is
    # noise-dominated and crosses zero.
    sigma = pygwb_reference["combined_sigma"]
    assert_close(omega / sigma, pygwb_reference["combined_point_estimate"] / sigma)
