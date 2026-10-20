"""The per-source ORF, the network response, and its shot-noise inflation.

The Monte Carlo checks compare against the analytic isotropic ORF in units of
the draws' own standard error, so they hold for any seed rather than at a tuned
tolerance.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.detector import (
    isotropic_overlap_reduction_function,
    load_sensitivity_map,
    network_response,
    overlap_reduction_function,
    projected_network_response,
    response_nodes,
    sample_network_response,
)
from astrogwb.inference import network_shot_noise_inflation

NUM_DRAWS = 40_000
#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0

FREQUENCIES = np.array([5.0, 30.0, 100.0, 300.0])
SEARCH_FREQUENCIES = np.geomspace(5.0, 1000.0, 120)


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


def _inflation(detectors: Sequence[str], seed: int) -> float:
    projected = projected_network_response(
        jax.random.key(seed),
        NUM_DRAWS,
        SEARCH_FREQUENCIES,
        detectors,
        load_sensitivity_map(detectors),
    )
    return float(network_shot_noise_inflation(projected))


@pytest.mark.parametrize(
    ("detectors", "lower", "upper"),
    [
        (["S1", "R1"], 1.15, 1.30),
        (["S2", "R2"], 1.40, 1.60),
        (["E1", "E2", "E3"], 1.10, 1.30),
    ],
)
def test_the_inflation_reproduces_the_monte_carlo_of_the_paper_networks(
    detectors: list[str], lower: float, upper: float
) -> None:
    assert lower < _inflation(detectors, seed=4) < upper


def test_a_misaligned_pair_inflates_more_than_an_aligned_one() -> None:
    assert 1.0 < _inflation(["S1", "R1"], seed=5) < _inflation(["S2", "R2"], seed=5)


def test_the_projected_response_has_unit_sample_mean() -> None:
    projected = projected_network_response(
        jax.random.key(6),
        1_000,
        SEARCH_FREQUENCIES,
        ["S1", "R1"],
        load_sensitivity_map(["S1", "R1"]),
    )

    assert float(jnp.mean(projected)) == pytest.approx(1.0, rel=1e-12)


def test_response_nodes_keep_the_mean_and_shrink_the_second_moment() -> None:
    draws = jax.random.gamma(jax.random.key(7), 2.0, (4_096,))

    nodes = response_nodes(draws, 16)

    assert nodes.shape == (16,)
    assert float(jnp.mean(nodes)) == pytest.approx(float(jnp.mean(draws)), rel=1e-12)
    assert float(jnp.mean(nodes**2)) <= float(jnp.mean(draws**2))
