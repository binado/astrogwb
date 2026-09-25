"""Per-bin Fisher matrices of the Gaussian spectrum likelihood."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    mock_merger_rate_fn,
    mock_target_model,
)
from jax.typing import ArrayLike
from numpyro.infer.util import log_density

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.importance.spectral import build_importance_spectrum
from astrogwb.populations import DEFAULT_DENSITY_SITES
from astrogwb.sampling import (
    SpectralDensityFn,
    derivative_cosine_matrix,
    fisher_eigenmodes,
    fisher_from_whitened_jacobian,
    fisher_matrix_per_bin,
    gwb_spectral_density_model,
    prior_sigma_along_modes,
    spectral_density_jacobian,
    whitened_jacobian,
)


@pytest.fixture
def bin_frequencies() -> jax.Array:
    """A short grid; named apart from conftest's 128-bin ``frequencies``."""
    return jnp.linspace(10.0, 100.0, 7)


@pytest.fixture
def scale(bin_frequencies: jax.Array) -> jax.Array:
    return 0.1 + 0.01 * bin_frequencies


@pytest.fixture
def power_law_params() -> dict[str, float]:
    return {"amplitude": 2.0, "index": 0.7, "offset": 0.3}


@pytest.fixture
def power_law(bin_frequencies: jax.Array) -> SpectralDensityFn:
    def spectrum(
        params: Mapping[str, ArrayLike],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        ratio = bin_frequencies / 10.0
        return jnp.asarray(
            params["amplitude"] * ratio ** params["index"] + params["offset"]
        ), {}

    return spectrum


def test_jacobian_and_fisher_match_the_closed_form(
    power_law: SpectralDensityFn,
    power_law_params: dict[str, float],
    bin_frequencies: jax.Array,
    scale: jax.Array,
) -> None:
    names = ("amplitude", "index")
    jacobian = spectral_density_jacobian(power_law, power_law_params, names)

    amplitude, index = power_law_params["amplitude"], power_law_params["index"]
    ratio = bin_frequencies / 10.0
    expected = jnp.stack(
        [ratio**index, amplitude * ratio**index * jnp.log(ratio)], axis=-1
    )
    np.testing.assert_allclose(jacobian, expected, rtol=1e-12)

    fisher = fisher_matrix_per_bin(power_law, power_law_params, names, scale=scale)
    assert fisher.shape == (bin_frequencies.size, 2, 2)
    np.testing.assert_allclose(
        fisher,
        expected[:, :, None] * expected[:, None, :] / scale[:, None, None] ** 2,
        rtol=1e-12,
    )


def test_columns_follow_the_requested_order_and_others_stay_fixed(
    power_law: SpectralDensityFn, power_law_params: dict[str, float]
) -> None:
    forward = spectral_density_jacobian(
        power_law, power_law_params, ("amplitude", "index")
    )
    reverse = spectral_density_jacobian(
        power_law, power_law_params, ("index", "amplitude")
    )
    np.testing.assert_array_equal(forward, reverse[:, ::-1])
    # An integer fiducial is promoted rather than breaking the derivative.
    integer = {**power_law_params, "amplitude": int(power_law_params["amplitude"])}
    np.testing.assert_allclose(
        spectral_density_jacobian(power_law, integer, ("amplitude",)),
        forward[:, :1],
    )


def test_an_unknown_parameter_is_rejected(
    power_law: SpectralDensityFn, power_law_params: dict[str, float]
) -> None:
    with pytest.raises(KeyError, match="missing"):
        spectral_density_jacobian(power_law, power_law_params, ("missing",))


def test_summed_fisher_is_the_hessian_of_the_model_likelihood(
    power_law: SpectralDensityFn,
    power_law_params: dict[str, float],
    scale: jax.Array,
) -> None:
    names = ("amplitude", "index", "offset")
    observed, _ = power_law(power_law_params)
    model = partial(
        gwb_spectral_density_model,
        spectral_density_fn=power_law,
        observed_spectral_density=observed,
        priors={},
        scale=scale,
    )

    def negative_log_likelihood(free: dict[str, jax.Array]) -> jax.Array:
        # The model samples nothing, so the parameters enter through the spectrum.
        def at_free(
            params: Mapping[str, ArrayLike],
        ) -> tuple[jax.Array, Mapping[str, ArrayLike]]:
            return power_law(free)

        conditioned = partial(model, spectral_density_fn=at_free)
        return -log_density(conditioned, (), {}, {})[0]

    free = {name: jnp.asarray(power_law_params[name]) for name in names}
    hessian = jax.hessian(negative_log_likelihood)(free)
    expected = jnp.array([[hessian[a][b] for b in names] for a in names])

    fisher = fisher_matrix_per_bin(power_law, power_law_params, names, scale=scale)
    np.testing.assert_allclose(fisher.sum(axis=0), expected, rtol=1e-10)


def test_masked_bins_contribute_zero_even_with_infinite_scale(
    power_law: SpectralDensityFn,
    power_law_params: dict[str, float],
    scale: jax.Array,
) -> None:
    names = ("amplitude", "index")
    mask = jnp.arange(scale.size) % 2 == 0
    masked = fisher_matrix_per_bin(
        power_law,
        power_law_params,
        names,
        scale=jnp.where(mask, scale, jnp.inf),
        frequency_mask=mask,
    )
    full = fisher_matrix_per_bin(power_law, power_law_params, names, scale=scale)
    assert np.all(np.isfinite(masked))
    np.testing.assert_array_equal(masked[~mask], 0.0)
    np.testing.assert_allclose(masked[mask], full[mask], rtol=1e-12)


def test_importance_jacobian_matches_finite_differences(
    mock_catalog_factory: Callable[..., PolarizationPowerCatalog],
) -> None:
    spectrum, _ = build_importance_spectrum(
        mock_catalog_factory(num_sources=256),
        source_model=mock_target_model(),
        merger_rate_fn=mock_merger_rate_fn(),
        density_sites=DEFAULT_DENSITY_SITES,
    )
    names = ("H0", "xi_0", "gamma")
    jacobian = spectral_density_jacobian(spectrum, FIDUCIALS, names)
    assert np.all(np.isfinite(jacobian))

    for column, name in enumerate(names):
        step = 1e-5 * abs(FIDUCIALS[name])
        upper = spectrum({**FIDUCIALS, name: FIDUCIALS[name] + step})[0]
        lower = spectrum({**FIDUCIALS, name: FIDUCIALS[name] - step})[0]
        finite_difference = (upper - lower) / (2.0 * step)
        np.testing.assert_allclose(
            jacobian[:, column],
            finite_difference,
            rtol=1e-6,
            atol=1e-8 * float(jnp.max(jnp.abs(finite_difference))),
            err_msg=name,
        )


# --------------------------------------------------------------------------- #
# Degeneracy views: whitened derivatives, cosines, eigenmodes
# --------------------------------------------------------------------------- #
def test_whitened_jacobian_is_the_derivative_in_noise_units(
    power_law: SpectralDensityFn,
    power_law_params: dict[str, float],
    bin_frequencies: jax.Array,
    scale: jax.Array,
) -> None:
    amplitude, index = power_law_params["amplitude"], power_law_params["index"]
    ratio = bin_frequencies / 10.0
    expected = (
        jnp.stack([ratio**index, amplitude * ratio**index * jnp.log(ratio)], axis=-1)
        / scale[:, None]
    )

    whitened = whitened_jacobian(
        power_law, power_law_params, ("amplitude", "index"), scale=scale
    )

    np.testing.assert_allclose(whitened, expected, rtol=1e-12)


def test_whitened_jacobian_masked_bins_are_zero_even_with_infinite_scale(
    power_law: SpectralDensityFn,
    power_law_params: dict[str, float],
    scale: jax.Array,
) -> None:
    mask = jnp.arange(scale.size) >= 2
    whitened = whitened_jacobian(
        power_law,
        power_law_params,
        ("amplitude", "index"),
        scale=jnp.where(mask, scale, jnp.inf),
        frequency_mask=mask,
    )

    np.testing.assert_array_equal(whitened[:2], 0.0)


def test_fisher_from_whitened_jacobian_matches_fisher_matrix_per_bin(
    power_law: SpectralDensityFn,
    power_law_params: dict[str, float],
    scale: jax.Array,
) -> None:
    names = ("amplitude", "index", "offset")
    whitened = whitened_jacobian(power_law, power_law_params, names, scale=scale)

    np.testing.assert_allclose(
        fisher_from_whitened_jacobian(whitened),
        fisher_matrix_per_bin(power_law, power_law_params, names, scale=scale),
        rtol=1e-12,
    )


def test_derivative_cosine_matrix_is_minus_the_two_parameter_correlation() -> None:
    fisher = np.array([[4.0, 3.0], [3.0, 9.0]])
    covariance = np.linalg.inv(fisher)
    correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])

    cosines = derivative_cosine_matrix(fisher)

    np.testing.assert_allclose(np.diag(cosines), 1.0)
    assert cosines[0, 1] == pytest.approx(-correlation)


def test_derivative_cosine_matrix_uninformed_parameter_is_nan() -> None:
    cosines = derivative_cosine_matrix(np.array([[4.0, 0.0], [0.0, 0.0]]))

    assert cosines[0, 0] == pytest.approx(1.0)
    assert np.all(np.isnan(cosines[1]))


def test_fisher_eigenmodes_diagonal_matrix_gives_scaled_axes() -> None:
    """Scales 2 and 0.5 turn diag(1, 64) into diag(4, 16); best-constrained first."""
    fisher = np.diag([1.0, 64.0])

    modes = fisher_eigenmodes(fisher, ("a", "b"), parameter_scales=(2.0, 0.5))

    np.testing.assert_allclose(modes.sigmas, [0.25, 0.5])
    np.testing.assert_allclose(modes.directions, [[0.0, 1.0], [1.0, 0.0]], atol=1e-12)


def test_fisher_eigenmodes_rotated_matrix_recovers_its_axes() -> None:
    angle = 0.3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    fisher = rotation @ np.diag([100.0, 1.0]) @ rotation.T

    modes = fisher_eigenmodes(fisher, ("a", "b"))

    np.testing.assert_allclose(modes.sigmas, [0.1, 1.0])
    np.testing.assert_allclose(modes.directions[:, 0], rotation[:, 0], atol=1e-12)


def test_fisher_eigenmodes_largest_component_is_positive() -> None:
    fisher = np.array([[2.0, -1.9], [-1.9, 2.0]])

    modes = fisher_eigenmodes(fisher, ("a", "b"))

    leading = modes.directions[np.argmax(np.abs(modes.directions), axis=0), [0, 1]]
    assert np.all(leading > 0.0)


def test_fisher_eigenmodes_do_not_depend_on_parameter_units() -> None:
    """Measuring `a` in units 1000 times smaller, with its scale to match."""
    fisher = np.array([[4.0, 1.5], [1.5, 1.0]])
    units = np.diag([1e-3, 1.0])

    original = fisher_eigenmodes(fisher, ("a", "b"), parameter_scales=(2.0, 1.0))
    converted = fisher_eigenmodes(
        units @ fisher @ units, ("a", "b"), parameter_scales=(2e3, 1.0)
    )

    np.testing.assert_allclose(converted.sigmas, original.sigmas, rtol=1e-12)
    np.testing.assert_allclose(converted.directions, original.directions, atol=1e-12)


def test_fisher_eigenmodes_singular_direction_is_unconstrained() -> None:
    fisher = np.array([[1.0, 1.0], [1.0, 1.0]])

    modes = fisher_eigenmodes(fisher, ("a", "b"))

    assert np.isfinite(modes.sigmas[0])
    assert modes.sigmas[1] == np.inf


@pytest.mark.parametrize(
    ("fisher", "scales"),
    [(np.eye(3), None), (np.eye(2), (1.0, 0.0)), (np.eye(2), (1.0,))],
    ids=["wrong-shape", "zero-scale", "short-scales"],
)
def test_fisher_eigenmodes_rejects_inconsistent_inputs(
    fisher: np.ndarray, scales: tuple[float, ...] | None
) -> None:
    with pytest.raises(ValueError, match="fisher has shape|parameter_scales"):
        fisher_eigenmodes(fisher, ("a", "b"), parameter_scales=scales)


def test_prior_sigma_along_modes_projects_onto_the_prior_axis() -> None:
    """Axis-aligned modes: the prior on `b` sits on its mode, in scaled units."""
    modes = fisher_eigenmodes(
        np.diag([100.0, 1.0]), ("a", "b"), parameter_scales=(1.0, 2.0)
    )

    widths = prior_sigma_along_modes(modes, {"b": 3.0})

    np.testing.assert_allclose(widths, [np.inf, 1.5])


def test_prior_sigma_along_modes_rejects_an_unknown_parameter() -> None:
    modes = fisher_eigenmodes(np.eye(2), ("a", "b"))

    with pytest.raises(KeyError, match="not in"):
        prior_sigma_along_modes(modes, {"c": 1.0})
