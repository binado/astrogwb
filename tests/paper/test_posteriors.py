"""The setup helpers shared by the posterior notebooks."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from repo import REPO_ROOT

from astrogwb.detector.config import load_detector_config
from astrogwb.paper.posteriors import (
    MAX_PAIRS,
    network_data,
    network_pair_data,
    shot_noise_variant,
)
from astrogwb.simulators import DetectorNoise, detector_noise

NETWORK = "ET-2L-aligned-CE-Hanford"


@dataclasses.dataclass(frozen=True)
class _Likelihood:
    """The one field ``shot_noise_variant`` replaces."""

    shot_noise: str | None


@pytest.fixture(scope="module")
def noise() -> DetectorNoise:
    """The fiducial network's noise on a grid straddling the band's edges."""
    registry = load_detector_config([REPO_ROOT / "config/detectors.toml"])
    frequencies = np.array([0.5, 1.0, 10.0, 50.0, 100.0, 5000.0])
    return detector_noise(frequencies, registry, 1.0, network_names=[NETWORK])


def test_network_data_masks_out_of_band_bins(noise: DetectorNoise) -> None:
    mask = np.asarray(
        network_data(noise, NETWORK, minimum_frequency=2.0, maximum_frequency=2048.0)[
            "frequency_mask"
        ]
    )

    np.testing.assert_array_equal(mask, [False, False, True, True, True, False])


def test_network_data_gives_masked_bins_unit_scale(noise: DetectorNoise) -> None:
    data = network_data(noise, NETWORK, minimum_frequency=2.0, maximum_frequency=2048.0)
    mask, scale = np.asarray(data["frequency_mask"]), np.asarray(data["scale"])

    np.testing.assert_array_equal(scale[~mask], 1.0)
    np.testing.assert_array_equal(scale[mask], noise.noise[NETWORK].scale[mask])


def test_network_pair_data_pads_to_max_pairs_with_masked_rows(
    noise: DetectorNoise,
) -> None:
    data = network_pair_data(
        noise, NETWORK, minimum_frequency=2.0, maximum_frequency=2048.0
    )
    mask = np.asarray(data["frequency_mask"])

    assert mask.shape == (MAX_PAIRS, noise.frequencies.size)
    assert not mask[3:].any()
    np.testing.assert_array_equal(np.asarray(data["pair_overlap"])[3:], 0.0)
    np.testing.assert_array_equal(np.asarray(data["scale"])[3:], 1.0)


def test_network_pair_data_keeps_the_noise_files_overlaps(
    noise: DetectorNoise,
) -> None:
    data = network_pair_data(
        noise, NETWORK, minimum_frequency=2.0, maximum_frequency=2048.0
    )

    np.testing.assert_array_equal(
        np.asarray(data["pair_overlap"])[:3], noise.noise[NETWORK].pair_overlap
    )


@pytest.mark.parametrize(
    ("variant", "mode", "has_variance"),
    [
        ("detector", None, False),
        ("amplitude", "amplitude", False),
        ("per_frequency", "per_frequency", False),
        ("modes", "modes", False),
        ("fixed", "fixed", True),
    ],
)
def test_shot_noise_variant_sets_the_mode(
    variant: str, mode: str | None, has_variance: bool
) -> None:
    likelihood, variance = shot_noise_variant(
        variant,
        _Likelihood(shot_noise="amplitude"),  # ty: ignore[invalid-argument-type]
        jnp.array([0.1, 0.2]),  # ty: ignore[invalid-argument-type]
    )
    assert likelihood.shot_noise == mode
    assert (variance is not None) is has_variance
