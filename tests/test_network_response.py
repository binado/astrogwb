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
from test_detector import _sky_quadrature_orf

from astrogwb.detector import (
    load_detector,
    overlap_reduction_function,
    pair_response,
    stokes_cross_power,
)
from astrogwb.waveform import quadrupolar_stokes_factors

NUM_DRAWS = 40_000
#: Standard errors a Monte Carlo mean may sit from its expectation.
MAX_STANDARD_ERRORS = 5.0

FREQUENCIES = np.array([5.0, 30.0, 100.0, 300.0])


def _isotropic_angles(key: jax.Array, num_draws: int) -> tuple[jax.Array, ...]:
    keys = jax.random.split(key, 4)
    return (
        jax.random.uniform(keys[1], (num_draws,), maxval=2.0 * jnp.pi),
        jnp.arcsin(jax.random.uniform(keys[0], (num_draws,), minval=-1.0, maxval=1.0)),
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
    [("H1", "L1"), ("E1", "E2")],
)
def test_the_isotropic_mean_of_the_per_source_orf_is_the_isotropic_orf(
    detector_1: str, detector_2: str
) -> None:
    first, second = load_detector(detector_1), load_detector(detector_2)
    draws = overlap_reduction_function(
        FREQUENCIES,
        first,
        second,
        *_isotropic_angles(jax.random.key(0), NUM_DRAWS),
    )
    expected = _sky_quadrature_orf(FREQUENCIES, first, second)

    assert draws.shape == (NUM_DRAWS, FREQUENCIES.size)
    assert np.all(_standard_errors(draws, expected) < MAX_STANDARD_ERRORS)


@pytest.mark.parametrize(
    ("names", "pairs"),
    [
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
    names: list[str], pairs: list[tuple[str, str]]
) -> None:
    angles = _isotropic_angles(jax.random.key(4), 100)

    stacked = pair_response(
        FREQUENCIES, [load_detector(name) for name in names], *angles
    )

    assert stacked.shape == (100, len(pairs), FREQUENCIES.size)
    for index, (first, second) in enumerate(pairs):
        np.testing.assert_array_equal(
            np.asarray(stacked[:, index]),
            np.asarray(
                overlap_reduction_function(
                    FREQUENCIES, load_detector(first), load_detector(second), *angles
                )
            ),
        )


def _random_stokes(key: jax.Array, num_draws: int) -> jax.Array:
    """Stokes parameters ``(4, num_draws, F)`` of random fully polarized sources."""
    inclination = jax.random.uniform(
        key, (num_draws, FREQUENCIES.size), minval=-1.0, maxval=1.0
    )
    return quadrupolar_stokes_factors(inclination)


def test_stokes_cross_power_of_a_detector_with_itself_is_blind_to_v() -> None:
    polar_cosine, azimuth, polarization, _ = _isotropic_angles(jax.random.key(5), 100)
    circular = jnp.array([0.0, 0.0, 0.0, 1.0])[:, None, None]

    h1 = load_detector("H1")
    power = stokes_cross_power(
        FREQUENCIES, h1, h1, polar_cosine, azimuth, polarization, circular
    )

    np.testing.assert_array_equal(np.asarray(power), 0.0)


def test_stokes_cross_power_of_a_detector_with_itself_is_non_negative() -> None:
    polar_cosine, azimuth, polarization, _ = _isotropic_angles(jax.random.key(6), 200)

    h1 = load_detector("H1")
    power = stokes_cross_power(
        FREQUENCIES,
        h1,
        h1,
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
    first, second = load_detector(detector_1), load_detector(detector_2)
    polar_cosine, azimuth, polarization, _ = _isotropic_angles(jax.random.key(9), 50)
    stokes = _random_stokes(jax.random.key(10), 50)
    flipped = stokes * jnp.array([1.0, -1.0, -1.0, 1.0])[:, None, None]

    rotated = stokes_cross_power(
        FREQUENCIES,
        first,
        second,
        polar_cosine,
        azimuth,
        polarization + jnp.pi / 4,
        stokes,
    )
    reflected = stokes_cross_power(
        FREQUENCIES,
        first,
        second,
        polar_cosine,
        azimuth,
        polarization,
        flipped,
    )

    np.testing.assert_allclose(
        np.asarray(rotated), np.asarray(reflected), atol=1e-12, rtol=1e-10
    )


@pytest.mark.parametrize("circular", [False, True])
def test_stokes_matches_direct_complex_strain_cross_power(circular: bool) -> None:
    pair = ("H1", "L1")
    ra = jnp.array([0.0, 1.0, 2.0, 2 * jnp.pi])[:, None]
    dec = jnp.array([-jnp.pi / 2, 0.4, jnp.pi / 2])[None, :]
    coordinates = {"right_ascension": ra, "declination": dec, "polarization_angle": 0.7}
    plus = jnp.array([1 + 2j, -0.4j, 0.6 - 1j, 1.0])
    cross = 1j * plus if circular else jnp.array([0.8 - 0.2j, -0.7 + 1j, 0.4j, -1j])
    product = jnp.conj(plus) * cross
    stokes = jnp.stack(
        [
            abs(plus) ** 2 + abs(cross) ** 2,
            abs(plus) ** 2 - abs(cross) ** 2,
            2 * jnp.real(product),
            2 * jnp.imag(product),
        ]
    )[:, None, None, :]
    first, second = (load_detector(name) for name in pair)
    a = first.project_polarizations_fd(FREQUENCIES, plus, cross, 1.3, **coordinates)
    b = second.project_polarizations_fd(FREQUENCIES, plus, cross, 1.3, **coordinates)
    direct = jnp.real(jnp.conj(a) * b)
    reduced = stokes_cross_power(
        FREQUENCIES, first, second, ra, dec, 0.7, stokes, gmst=1.3
    )
    # A forward error bound for the two equivalent contractions, including
    # cancellation near antenna zeros (scaled by the unprojected power).
    bound = 128 * np.finfo(np.float64).eps * np.asarray(stokes[0])
    assert np.all(np.abs(reduced - direct) <= bound)


@pytest.mark.parametrize("pair", [("H1", "L1"), ("E1", "E2"), ("S2", "R2")])
def test_native_unpolarized_average_matches_independent_geometry_quadrature(
    pair: tuple[str, str],
) -> None:
    first, second = (load_detector(name) for name in pair)
    mu, weights = np.polynomial.legendre.leggauss(48)
    phi = 2 * np.pi * np.arange(128) / 128
    frequencies = np.array([0.0, 10.0, 50.0])
    response = 5 * stokes_cross_power(
        frequencies,
        first,
        second,
        phi[None, :],
        np.arcsin(mu[:, None]),
        0.7,
        np.array([1.0, 0.0, 0.0, 0.0])[:, None, None, None],
        gmst=1.2,
    )
    integrated = np.sum(np.asarray(response) * weights[:, None, None], axis=(0, 1)) / (
        2 * phi.size
    )
    expected = _sky_quadrature_orf(frequencies, first, second)
    # Both independent spherical quadratures are converged at these frequencies.
    assert np.all(np.abs(integrated - expected) < 512 * np.finfo(np.float64).eps)
