"""The per-source ORF and the network response.

The Monte Carlo checks compare against the analytic isotropic ORF in units of
the draws' own standard error, so they hold for any seed rather than at a tuned
tolerance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.detector import (
    isotropic_overlap_reduction_function,
    load_sensitivity_map,
    network_response,
    overlap_reduction_function,
    pair_response,
    pair_terms,
    sample_network_response,
    sample_pair_response,
    stokes_cross_power,
)
from astrogwb.waveform import polarization_stokes

NUM_DRAWS = 40_000
#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0

FREQUENCIES = np.array([5.0, 30.0, 100.0, 300.0])


def _isotropic_angles(key: jax.Array, num_draws: int) -> tuple[jax.Array, ...]:
    keys = jax.random.split(key, 4)
    return (
        jax.random.uniform(keys[0], (num_draws,), minval=-1.0, maxval=1.0),
        jax.random.uniform(keys[1], (num_draws,), maxval=2.0 * jnp.pi),
        jax.random.uniform(keys[2], (num_draws,), maxval=jnp.pi),
        jax.random.uniform(keys[3], (num_draws,), minval=-1.0, maxval=1.0),
    )


def _standard_errors(draws: jax.Array, expected: np.ndarray) -> np.ndarray:
    """How many standard errors the mean of ``draws`` sits from ``expected``."""
    values = np.asarray(draws)
    error = values.std(axis=0) / np.sqrt(values.shape[0])
    return np.abs(values.mean(axis=0) - expected) / error


@pytest.mark.parametrize(
    ("detector_1", "detector_2"),
    [("H1", "L1"), ("E1", "E2"), ("S1", "R1"), ("S2", "R2"), ("C1", "S1")],
)
def test_the_isotropic_mean_of_the_per_source_orf_is_the_isotropic_orf(
    detector_1: str, detector_2: str
) -> None:
    draws = overlap_reduction_function(
        FREQUENCIES,
        detector_1,
        detector_2,
        *_isotropic_angles(jax.random.key(0), NUM_DRAWS),
    )
    expected = isotropic_overlap_reduction_function(FREQUENCIES, detector_1, detector_2)

    assert draws.shape == (NUM_DRAWS, FREQUENCIES.size)
    assert np.all(_standard_errors(draws, expected) < MAX_STANDARD_ERRORS)


def test_a_detector_paired_with_itself_sees_a_non_negative_power() -> None:
    draws = overlap_reduction_function(
        FREQUENCIES, "H1", "H1", *_isotropic_angles(jax.random.key(1), 1_000)
    )

    assert np.all(np.asarray(draws) >= 0.0)
    np.testing.assert_allclose(
        np.asarray(draws), np.broadcast_to(np.asarray(draws[:, :1]), draws.shape)
    )


def test_the_per_source_orf_broadcasts_over_the_angles() -> None:
    polar_cosine = jnp.linspace(-1.0, 1.0, 5)[:, None]
    azimuth = jnp.linspace(0.0, 6.0, 3)[None, :]

    draws = overlap_reduction_function(
        FREQUENCIES, "H1", "L1", polar_cosine, azimuth, 0.3, 0.5
    )

    assert draws.shape == (5, 3, FREQUENCIES.size)


@pytest.mark.parametrize("detectors", [["S1", "R1"], ["E1", "E2", "E3", "C1"]])
def test_the_network_response_has_unit_isotropic_mean(detectors: list[str]) -> None:
    draws = sample_network_response(
        jax.random.key(2),
        NUM_DRAWS,
        FREQUENCIES,
        detectors,
        load_sensitivity_map(detectors),
    )

    assert np.all(
        _standard_errors(draws, np.ones(FREQUENCIES.size)) < MAX_STANDARD_ERRORS
    )


def test_a_single_pair_response_does_not_depend_on_the_noise() -> None:
    angles = _isotropic_angles(jax.random.key(3), 100)
    sensitivities = load_sensitivity_map(["S1", "R1"])

    response = network_response(FREQUENCIES, ["S1", "R1"], sensitivities, *angles)
    orf = overlap_reduction_function(FREQUENCIES, "S1", "R1", *angles)

    np.testing.assert_allclose(
        np.asarray(response),
        np.asarray(orf / isotropic_overlap_reduction_function(FREQUENCIES, "S1", "R1")),
    )


def test_a_misaligned_pair_scatters_more_than_an_aligned_one() -> None:
    """A nearly cancelled isotropic ORF leaves sources of either sign: more scatter."""

    def variance(detectors: list[str]) -> np.ndarray:
        draws = sample_network_response(
            jax.random.key(8),
            NUM_DRAWS,
            FREQUENCIES,
            detectors,
            load_sensitivity_map(detectors),
        )
        return np.asarray(jnp.var(draws, axis=0))

    assert np.all(variance(["S1", "R1"]) < variance(["S2", "R2"]))


@pytest.mark.parametrize(
    ("detectors", "pairs"),
    [
        (["S1", "R1"], [("S1", "R1")]),
        (
            ["E1", "E2", "E3", "C1"],
            [
                ("E1", "E2"),
                ("E1", "E3"),
                ("E1", "C1"),
                ("E2", "E3"),
                ("E2", "C1"),
                ("E3", "C1"),
            ],
        ),
    ],
)
def test_pair_response_stacks_each_pairs_orf_in_a_less_than_b_order(
    detectors: list[str], pairs: list[tuple[str, str]]
) -> None:
    angles = _isotropic_angles(jax.random.key(4), 100)

    stacked = pair_response(FREQUENCIES, detectors, *angles)

    assert stacked.shape == (100, len(pairs), FREQUENCIES.size)
    for index, (first, second) in enumerate(pairs):
        np.testing.assert_array_equal(
            np.asarray(stacked[:, index]),
            np.asarray(overlap_reduction_function(FREQUENCIES, first, second, *angles)),
        )


def test_pair_response_draws_combine_to_the_network_response_draws() -> None:
    """Same key, same angles: the optimal combination of the pairs is the network."""
    detectors = ["E1", "E2", "E3", "C1"]
    sensitivities = load_sensitivity_map(detectors)
    terms = pair_terms(FREQUENCIES, detectors, sensitivities)
    weights = np.stack([term.weight for term in terms])
    denominator = np.sum([term.weight * term.overlap for term in terms], axis=0)

    pairs = sample_pair_response(jax.random.key(5), 200, FREQUENCIES, detectors)
    network = sample_network_response(
        jax.random.key(5), 200, FREQUENCIES, detectors, sensitivities
    )

    np.testing.assert_allclose(
        np.sum(weights * np.asarray(pairs), axis=1) / denominator,
        np.asarray(network),
        rtol=1e-12,
    )


def test_pair_response_has_each_pairs_isotropic_orf_as_its_mean() -> None:
    detectors = ["S2", "R2", "C1"]
    draws = sample_pair_response(jax.random.key(6), NUM_DRAWS, FREQUENCIES, detectors)
    expected = np.stack(
        [
            isotropic_overlap_reduction_function(FREQUENCIES, first, second)
            for first, second in [("S2", "R2"), ("S2", "C1"), ("R2", "C1")]
        ]
    )

    assert np.all(_standard_errors(draws, expected) < MAX_STANDARD_ERRORS)


def _random_stokes(key: jax.Array, num_draws: int) -> jax.Array:
    """Stokes parameters ``(4, num_draws, F)`` of random fully polarized sources."""
    keys = jax.random.split(key, 4)
    shape = (num_draws, FREQUENCIES.size)
    plus = jax.random.normal(keys[0], shape) + 1j * jax.random.normal(keys[1], shape)
    cross = jax.random.normal(keys[2], shape) + 1j * jax.random.normal(keys[3], shape)
    return jnp.swapaxes(polarization_stokes(plus, cross), 1, 2)


@pytest.mark.parametrize(
    ("detector_1", "detector_2"), [("H1", "L1"), ("E1", "E2"), ("C1", "S1")]
)
def test_stokes_cross_power_of_unpolarized_sources_averages_to_the_isotropic_orf(
    detector_1: str, detector_2: str
) -> None:
    polar_cosine, azimuth, polarization, _ = _isotropic_angles(
        jax.random.key(4), NUM_DRAWS
    )
    unpolarized = jnp.array([1.0, 0.0, 0.0, 0.0])[:, None, None]

    draws = 5.0 * stokes_cross_power(
        FREQUENCIES,
        detector_1,
        detector_2,
        polar_cosine,
        azimuth,
        polarization,
        unpolarized,
    )
    expected = isotropic_overlap_reduction_function(FREQUENCIES, detector_1, detector_2)

    assert np.all(_standard_errors(draws, expected) < MAX_STANDARD_ERRORS)


def test_stokes_cross_power_of_a_detector_with_itself_is_blind_to_v() -> None:
    polar_cosine, azimuth, polarization, _ = _isotropic_angles(jax.random.key(5), 100)
    circular = jnp.array([0.0, 0.0, 0.0, 1.0])[:, None, None]

    power = stokes_cross_power(
        FREQUENCIES, "H1", "H1", polar_cosine, azimuth, polarization, circular
    )

    np.testing.assert_array_equal(np.asarray(power), 0.0)


def test_stokes_cross_power_of_a_detector_with_itself_is_non_negative() -> None:
    polar_cosine, azimuth, polarization, _ = _isotropic_angles(jax.random.key(6), 200)

    power = stokes_cross_power(
        FREQUENCIES,
        "H1",
        "H1",
        polar_cosine,
        azimuth,
        polarization,
        _random_stokes(jax.random.key(7), 200),
    )

    assert np.all(np.asarray(power) >= -1e-12 * np.abs(np.asarray(power)).max())


@pytest.mark.parametrize(("detector_1", "detector_2"), [("H1", "L1"), ("S2", "R2")])
def test_stokes_cross_power_rotates_linear_polarization_by_four_psi(
    detector_1: str, detector_2: str
) -> None:
    """Turning the basis by pi/4 is the same as flipping the signs of Q and U."""
    polar_cosine, azimuth, polarization, _ = _isotropic_angles(jax.random.key(9), 50)
    stokes = _random_stokes(jax.random.key(10), 50)
    flipped = stokes * jnp.array([1.0, -1.0, -1.0, 1.0])[:, None, None]

    rotated = stokes_cross_power(
        FREQUENCIES,
        detector_1,
        detector_2,
        polar_cosine,
        azimuth,
        polarization + jnp.pi / 4,
        stokes,
    )
    reflected = stokes_cross_power(
        FREQUENCIES,
        detector_1,
        detector_2,
        polar_cosine,
        azimuth,
        polarization,
        flipped,
    )

    np.testing.assert_allclose(
        np.asarray(rotated), np.asarray(reflected), atol=1e-12, rtol=1e-10
    )
