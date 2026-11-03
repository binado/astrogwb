"""The setup helpers shared by the posterior notebooks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from repo import REPO_ROOT

from astrogwb.detector import (
    effective_psd,
    gaussian_bin_scale,
    pair_bin_scale,
    pair_terms,
)
from astrogwb.detector.config import load_detector_config
from astrogwb.paper.posteriors import network_data, pair_network
from astrogwb.simulators import NetworkNoise

NETWORK = "ET-2L-aligned-CE-Hanford"


@pytest.fixture(scope="module")
def frequencies() -> np.ndarray:
    """A grid straddling the band's edges."""
    return np.array([0.5, 1.0, 10.0, 50.0, 100.0, 5000.0])


@pytest.fixture(scope="module")
def noise(frequencies: np.ndarray) -> NetworkNoise:
    """The fiducial network's noise on ``frequencies``."""
    registry = load_detector_config([REPO_ROOT / "config/detectors.toml"])
    detectors = registry.build_network(NETWORK)
    psd = effective_psd(frequencies, detectors)
    return NetworkNoise(
        pair_overlap=np.stack(
            [term.overlap for term in pair_terms(frequencies, detectors)]
        ),
        pair_scale=np.asarray(pair_bin_scale(frequencies, detectors, 1.0)),
        effective_psd=np.asarray(psd),
        scale=np.asarray(gaussian_bin_scale(jnp.asarray(psd), 1.0, frequencies)),
    )


def test_network_data_masks_out_of_band_bins(
    frequencies: np.ndarray, noise: NetworkNoise
) -> None:
    mask = np.asarray(
        network_data(
            frequencies, noise, minimum_frequency=2.0, maximum_frequency=2048.0
        )["frequency_mask"]
    )

    np.testing.assert_array_equal(mask, [False, False, True, True, True, False])


def test_network_data_gives_masked_bins_unit_scale(
    frequencies: np.ndarray, noise: NetworkNoise
) -> None:
    data = network_data(
        frequencies, noise, minimum_frequency=2.0, maximum_frequency=2048.0
    )
    mask, scale = np.asarray(data["frequency_mask"]), np.asarray(data["scale"])

    np.testing.assert_array_equal(scale[~mask], 1.0)
    np.testing.assert_array_equal(scale[mask], noise.scale[mask])


def test_pair_network_counts_only_in_band_finite_entries(
    frequencies: np.ndarray, noise: NetworkNoise
) -> None:
    network = pair_network(
        frequencies, noise, minimum_frequency=2.0, maximum_frequency=2048.0
    )
    in_band = np.array([False, False, True, True, True, False])
    counted = in_band & np.isfinite(noise.pair_scale) & (noise.pair_scale > 0.0)

    np.testing.assert_array_equal(np.asarray(network.keep), np.flatnonzero(counted))


def test_pair_network_keeps_the_networks_overlaps(
    frequencies: np.ndarray, noise: NetworkNoise
) -> None:
    network = pair_network(
        frequencies, noise, minimum_frequency=2.0, maximum_frequency=2048.0
    )

    np.testing.assert_array_equal(
        np.asarray(network.pair_overlap),
        noise.pair_overlap.reshape(-1)[np.asarray(network.keep)],
    )
