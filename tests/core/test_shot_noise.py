"""The rank-one shot-noise likelihood and its amplitude scatter."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from astrogwb.inference.shot_noise import (
    amplitude_direction,
    amplitude_shot_noise_variance,
    low_rank_gaussian_log_likelihood,
    pad_modes,
    per_frequency_direction,
    rank_one_gaussian_log_likelihood,
    shot_noise_modes,
)


@pytest.fixture
def template() -> np.ndarray:
    return np.array([3.0, 2.0, 1.2, 0.6, 0.2])


@pytest.fixture
def observed() -> np.ndarray:
    return np.array([3.3, 1.8, 1.5, 0.4, 0.3])


@pytest.fixture
def scale() -> np.ndarray:
    return np.array([0.4, 0.3, 0.5, 0.2, 0.6])


def _dense(
    observed: np.ndarray, prediction: np.ndarray, scale: np.ndarray, u: np.ndarray
) -> float:
    covariance = np.diag(scale**2) + np.outer(u, u)
    return float(
        dist.MultivariateNormal(
            jnp.asarray(prediction), jnp.asarray(covariance)
        ).log_prob(jnp.asarray(observed))
    )


def test_rank_one_log_likelihood_matches_the_dense_gaussian(
    template: np.ndarray, observed: np.ndarray, scale: np.ndarray
) -> None:
    direction = np.array([0.5, -0.2, 0.3, 0.1, 0.4])
    value = rank_one_gaussian_log_likelihood(observed, template, scale, direction)

    assert float(value) == pytest.approx(
        _dense(observed, template, scale, direction), rel=1e-12
    )


def test_rank_one_log_likelihood_with_a_mask_matches_the_compressed_dense_gaussian(
    template: np.ndarray, observed: np.ndarray, scale: np.ndarray
) -> None:
    direction = np.array([0.5, -0.2, 0.3, 0.1, 0.4])
    keep = np.array([True, False, True, True, False])
    infinite = np.where(keep, scale, np.inf)

    value = rank_one_gaussian_log_likelihood(
        observed, template, infinite, direction, keep
    )

    expected = _dense(observed[keep], template[keep], scale[keep], direction[keep])
    assert float(value) == pytest.approx(expected, rel=1e-12)


def test_rank_one_log_likelihood_with_zero_direction_is_the_diagonal_gaussian(
    template: np.ndarray, observed: np.ndarray, scale: np.ndarray
) -> None:
    value = rank_one_gaussian_log_likelihood(
        observed, template, scale, np.zeros_like(template)
    )
    diagonal = dist.Normal(jnp.asarray(template), jnp.asarray(scale)).log_prob(
        jnp.asarray(observed)
    )

    assert float(value) == pytest.approx(float(diagonal.sum()), rel=1e-14)


def test_rank_one_log_likelihood_stays_finite_at_detector_scales(
    template: np.ndarray, observed: np.ndarray, scale: np.ndarray
) -> None:
    """A GWB spectrum and its noise near 1e-48: whitening keeps it finite."""
    tiny = 1e-48
    value = rank_one_gaussian_log_likelihood(
        observed * tiny, template * tiny, scale * tiny, 0.1 * template * tiny
    )
    shift = -template.size * np.log(tiny)

    assert float(value) == pytest.approx(
        _dense(observed, template, scale, 0.1 * template) + shift, rel=1e-12
    )


@pytest.mark.parametrize("relative_variance", [1e-3, 4e-2])
def test_amplitude_direction_gives_the_amplitude_variance_with_shot_noise(
    template: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    relative_variance: float,
) -> None:
    r"""Along the template, :math:`V(A) = \rho^{-2} + A^2 s^2` and a :math:`\ln V`."""
    amplitudes = np.linspace(0.8, 1.3, 6)
    log_likelihood = np.array(
        [
            rank_one_gaussian_log_likelihood(
                observed,
                a * template,
                scale,
                amplitude_direction(a * template, relative_variance),
            )
            for a in amplitudes
        ]
    )
    snr_squared = np.sum(template**2 / scale**2)
    estimate = np.sum(observed * template / scale**2) / snr_squared
    variance = 1.0 / snr_squared + amplitudes**2 * relative_variance
    marginal = -0.5 * ((estimate - amplitudes) ** 2 / variance + np.log(variance))

    np.testing.assert_allclose(
        log_likelihood - log_likelihood[0], marginal - marginal[0], rtol=1e-10
    )


@pytest.mark.parametrize("relative_sd", [0.002, 0.05])
def test_amplitude_shot_noise_variance_of_a_flat_relative_scatter_is_its_square(
    template: np.ndarray, scale: np.ndarray, relative_sd: float
) -> None:
    variance = (relative_sd * template) ** 2

    assert float(
        amplitude_shot_noise_variance(template, variance, scale)
    ) == pytest.approx(relative_sd**2, rel=1e-12)


def test_amplitude_shot_noise_variance_ignores_masked_and_empty_bins(
    template: np.ndarray, scale: np.ndarray
) -> None:
    spectrum = template.copy()
    spectrum[-1] = 0.0
    variance = (0.01 * spectrum) ** 2
    variance[1] = 1e6
    keep = np.array([True, False, True, True, True])

    assert float(
        amplitude_shot_noise_variance(spectrum, variance, scale, keep)
    ) == pytest.approx(1e-4, rel=1e-12)


def test_per_frequency_direction_of_a_flat_relative_scatter_is_the_amplitude_one(
    template: np.ndarray,
) -> None:
    np.testing.assert_allclose(
        per_frequency_direction((0.03 * template) ** 2),
        amplitude_direction(template, 0.03**2),
        rtol=1e-14,
    )


def _whitened_dense(
    observed: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    modes: np.ndarray,
    block: np.ndarray,
) -> float:
    root = np.diag(scale)
    covariance = root @ (np.eye(scale.size) + modes @ block @ modes.T) @ root
    return float(
        dist.MultivariateNormal(
            jnp.asarray(prediction), jnp.asarray(covariance)
        ).log_prob(jnp.asarray(observed))
    )


@pytest.fixture
def modes() -> np.ndarray:
    """Two orthonormal columns over five bins."""
    rng = np.random.default_rng(1)
    return np.linalg.qr(rng.normal(size=(5, 2)))[0]


@pytest.fixture
def block() -> np.ndarray:
    return np.array([[2.0, 0.3], [0.3, 0.5]])


def test_low_rank_log_likelihood_matches_the_dense_gaussian(
    template: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    modes: np.ndarray,
    block: np.ndarray,
) -> None:
    value = low_rank_gaussian_log_likelihood(observed, template, scale, modes, block)

    assert float(value) == pytest.approx(
        _whitened_dense(observed, template, scale, modes, block), rel=1e-12
    )


def test_low_rank_log_likelihood_of_one_mode_is_the_rank_one_one(
    template: np.ndarray, observed: np.ndarray, scale: np.ndarray
) -> None:
    direction = np.array([0.5, -0.2, 0.3, 0.1, 0.4])
    whitened = direction / scale
    norm = np.linalg.norm(whitened)

    value = low_rank_gaussian_log_likelihood(
        observed, template, scale, (whitened / norm)[:, None], np.array([[norm**2]])
    )

    assert float(value) == pytest.approx(
        float(rank_one_gaussian_log_likelihood(observed, template, scale, direction)),
        rel=1e-12,
    )


def test_zero_padded_modes_change_nothing(
    template: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    modes: np.ndarray,
    block: np.ndarray,
) -> None:
    padded_block = np.pad(block, ((0, 3), (0, 3)))

    padded = low_rank_gaussian_log_likelihood(
        observed, template, scale, pad_modes(modes, 5), padded_block
    )

    assert float(padded) == pytest.approx(
        float(
            low_rank_gaussian_log_likelihood(observed, template, scale, modes, block)
        ),
        rel=1e-14,
    )


def test_low_rank_log_likelihood_drops_masked_bins_with_infinite_scale(
    template: np.ndarray,
    observed: np.ndarray,
    scale: np.ndarray,
    block: np.ndarray,
) -> None:
    keep = np.array([True, False, True, True, False])
    kept_modes = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 2)))[0]
    modes = np.zeros((5, 2))
    modes[keep] = kept_modes
    infinite = np.where(keep, scale, np.inf)

    value = low_rank_gaussian_log_likelihood(
        observed, template, infinite, modes, block, keep
    )

    assert np.isfinite(float(value))
    assert float(value) == pytest.approx(
        _whitened_dense(observed[keep], template[keep], scale[keep], kept_modes, block),
        rel=1e-12,
    )


@pytest.fixture
def spectrum_covariance() -> np.ndarray:
    """A covariance whose whitened eigenvalues are 1, 1e-3 and 1e-9 times smaller."""
    rng = np.random.default_rng(3)
    basis = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    return basis @ np.diag([1.0, 1e-3, 1e-9, 1e-12]) @ basis.T


@pytest.mark.parametrize(
    ("tolerance", "max_modes", "expected"),
    [(1e-6, 32, 2), (1e-10, 32, 3), (1e-6, 1, 1)],
)
def test_shot_noise_modes_keep_the_trace_fraction_up_to_the_cap(
    spectrum_covariance: np.ndarray, tolerance: float, max_modes: int, expected: int
) -> None:
    modes = shot_noise_modes(
        spectrum_covariance, np.ones(4), tolerance=tolerance, max_modes=max_modes
    )

    assert modes.shape == (4, expected)
    np.testing.assert_allclose(
        np.asarray(modes.T @ modes), np.eye(expected), atol=1e-12
    )


def test_shot_noise_modes_are_zero_in_masked_bins(
    spectrum_covariance: np.ndarray,
) -> None:
    mask = np.array([True, True, False, True])

    modes = shot_noise_modes(spectrum_covariance, np.ones(4), mask)

    assert np.all(np.asarray(modes)[~mask] == 0.0)
