from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.constants import (
    GPC_IN_METERS,
    GRAVITATIONAL_CONSTANT,
    ISCO_ALPHA,
    SECONDS_PER_YEAR,
    SOLAR_MASS_IN_KILOGRAMS,
    SOLAR_MASS_IN_SECONDS,
    SPEED_OF_LIGHT,
)
from astrogwb.cosmology import hubble_constant_si
from astrogwb.gwb import (
    analytic_spectral_density,
    analytic_spectral_density_from_mass_moments,
    omega_gw_from_spectral_density,
    precompute_cumulative_mass_moments,
    uniform_prior_mass_moments,
)
from astrogwb.gwb.analytic import (
    _cumulative_mass_moment_grid,
    _uniform_cumulative_mass_moment,
)
from numpy.polynomial.legendre import leggauss

jax.config.update("jax_enable_x64", True)

HYPERPARAMETERS = {
    "H0": 70.0,
    "Omega_m": 0.0,
    "rate": 25.0,
}
Z_MIN = 0.0
Z_MAX = 2.0
MASS_MIN = 5.0
MASS_MAX = 45.0
MASS_RANGE = MASS_MAX - MASS_MIN


def constant_rate(redshift: jax.Array, hyperparameters: Mapping[str, Any]) -> jax.Array:
    return jnp.full_like(redshift, hyperparameters["rate"])


def uniform_joint_mass(
    mass_1: jax.Array,
    mass_2: jax.Array,
    hyperparameters: Mapping[str, Any],
) -> jax.Array:
    del mass_2, hyperparameters
    return jnp.full_like(mass_1, 2.0 / MASS_RANGE**2)


def _analytic(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, Any] = HYPERPARAMETERS,
    *,
    alpha: float = math.inf,
    quadrature_order: int = 64,
) -> jax.Array:
    return analytic_spectral_density(
        frequencies,
        hyperparameters,
        constant_rate,
        uniform_joint_mass,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        minimum_component_mass=MASS_MIN,
        maximum_component_mass=MASS_MAX,
        alpha=alpha,
        quadrature_order=quadrature_order,
    )


def _strain_coefficient(h0: float) -> float:
    return (
        2.0
        * (GRAVITATIONAL_CONSTANT * SOLAR_MASS_IN_KILOGRAMS) ** (5.0 / 3.0)
        / (
            3.0
            * np.pi ** (1.0 / 3.0)
            * SPEED_OF_LIGHT**2
            * hubble_constant_si(h0)
            * GPC_IN_METERS**3
            * SECONDS_PER_YEAR
        )
    )


def _mapped_legendre(
    order: int, lower: float | np.ndarray, upper: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(order)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return (
        midpoint[..., None] + half_width[..., None] * nodes,
        half_width[..., None] * weights,
    )


def _direct_uniform_mass_moment(total_mass_cutoff: float, order: int) -> float:
    """Integrate directly over the cutoff ordered ``(m1, m2)`` triangle."""
    if total_mass_cutoff <= 2.0 * MASS_MIN:
        return 0.0

    cutoff = min(total_mass_cutoff, 2.0 * MASS_MAX)
    density = 2.0 / MASS_RANGE**2

    def integrate_segment(
        mass_1_lower: float,
        mass_1_upper: float,
        mass_2_upper_fn: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        if mass_1_upper <= mass_1_lower:
            return 0.0
        mass_1, mass_1_weights = _mapped_legendre(order, mass_1_lower, mass_1_upper)
        mass_2, mass_2_weights = _mapped_legendre(
            order, MASS_MIN, mass_2_upper_fn(mass_1)
        )
        mass_1_grid = mass_1[:, None]
        chirp_mass_power = mass_1_grid * mass_2 / (mass_1_grid + mass_2) ** (1.0 / 3.0)
        return float(
            np.sum(
                mass_1_weights[:, None] * mass_2_weights * density * chirp_mass_power
            )
        )

    lower_segment = integrate_segment(
        MASS_MIN,
        min(MASS_MAX, 0.5 * cutoff),
        lambda mass_1: mass_1,
    )
    upper_segment = integrate_segment(
        max(MASS_MIN, 0.5 * cutoff),
        min(MASS_MAX, cutoff - MASS_MIN),
        lambda mass_1: cutoff - mass_1,
    )
    return lower_segment + upper_segment


def _direct_uniform_spectral_density(
    frequencies: np.ndarray, *, alpha: float, order: int
) -> np.ndarray:
    redshift, redshift_weights = _mapped_legendre(order, Z_MIN, Z_MAX)
    values = []
    for frequency in frequencies:
        mass_moments = np.asarray(
            [
                _direct_uniform_mass_moment(
                    alpha / (frequency * (1.0 + value) * SOLAR_MASS_IN_SECONDS),
                    order,
                )
                for value in redshift
            ]
        )
        population_moment = np.sum(
            redshift_weights
            * HYPERPARAMETERS["rate"]
            / (1.0 + redshift) ** (4.0 / 3.0)
            * mass_moments
        )
        values.append(
            _strain_coefficient(HYPERPARAMETERS["H0"])
            * frequency ** (-7.0 / 3.0)
            * population_moment
        )
    return np.asarray(values)


def test_cumulative_mass_moment_matches_direct_component_mass_integral() -> None:
    total_mass, cumulative = _cumulative_mass_moment_grid(
        {},
        uniform_joint_mass,
        minimum_component_mass=MASS_MIN,
        maximum_component_mass=MASS_MAX,
        mass_ratio_quadrature_order=64,
        n_interp_grid=2049,
    )
    cutoffs = np.array(
        [
            2.0 * MASS_MIN - 1.0,
            2.0 * MASS_MIN,
            37.0,
            MASS_MIN + MASS_MAX,
            73.0,
            2.0 * MASS_MAX,
            np.inf,
        ]
    )

    actual = np.asarray(jnp.interp(jnp.asarray(cutoffs), total_mass, cumulative))
    expected = np.asarray(
        [_direct_uniform_mass_moment(cutoff, order=128) for cutoff in cutoffs]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-8)
    assert actual[0] == 0.0
    assert actual[1] == 0.0
    assert actual[-1] == actual[-2]


def test_mass_prior_is_called_once_only_inside_the_physical_triangle() -> None:
    calls: list[tuple[int, ...]] = []

    def guarded_prior(
        mass_1: jax.Array,
        mass_2: jax.Array,
        hyperparameters: Mapping[str, Any],
    ) -> jax.Array:
        del hyperparameters
        calls.append(mass_1.shape)
        valid = (mass_2 >= MASS_MIN) & (mass_1 >= mass_2) & (mass_1 <= MASS_MAX)
        return jnp.where(valid, 2.0 / MASS_RANGE**2, jnp.nan)

    _, cumulative = _cumulative_mass_moment_grid(
        {},
        guarded_prior,
        minimum_component_mass=MASS_MIN,
        maximum_component_mass=MASS_MAX,
        mass_ratio_quadrature_order=12,
        n_interp_grid=9,
    )

    assert calls == [(9, 12)]
    assert np.all(np.isfinite(np.asarray(cumulative)))


def test_featured_mass_population_converges_when_refining_grid() -> None:
    normalization_nodes, normalization_weights = _mapped_legendre(
        256, MASS_MIN, MASS_MAX
    )
    feature = 1.0 + 5.0 * np.exp(-0.5 * ((normalization_nodes - 35.0) / 2.5) ** 2)
    normalization = np.sum(
        normalization_weights * feature * (normalization_nodes - MASS_MIN)
    )

    def featured_prior(
        mass_1: jax.Array,
        mass_2: jax.Array,
        hyperparameters: Mapping[str, Any],
    ) -> jax.Array:
        del mass_2, hyperparameters
        primary_feature = 1.0 + 5.0 * jnp.exp(-0.5 * ((mass_1 - 35.0) / 2.5) ** 2)
        return primary_feature / normalization

    frequencies = jnp.array([20.0, 40.0, 80.0, 120.0])

    def evaluate(n_interp_grid: int) -> jax.Array:
        return analytic_spectral_density(
            frequencies,
            HYPERPARAMETERS,
            constant_rate,
            featured_prior,
            minimum_redshift=Z_MIN,
            maximum_redshift=Z_MAX,
            minimum_component_mass=MASS_MIN,
            maximum_component_mass=MASS_MAX,
            alpha=ISCO_ALPHA,
            quadrature_order=64,
            n_interp_grid=n_interp_grid,
        )

    coarse = np.asarray(evaluate(1025))
    default = np.asarray(evaluate(2049))
    refined = np.asarray(evaluate(4097))

    assert np.max(np.abs(default / refined - 1.0)) < np.max(
        np.abs(coarse / refined - 1.0)
    )
    np.testing.assert_allclose(
        default,
        refined,
        rtol=1e-4,
    )


def test_cumulative_mass_parameter_gradient_matches_finite_difference() -> None:
    queries = jnp.array([25.0, 50.0, 75.0])

    def evaluate(tilt: jax.Array) -> jax.Array:
        normalization = 0.5 * MASS_RANGE**2 + tilt * MASS_RANGE**3 / 6.0

        def tilted_prior(
            mass_1: jax.Array,
            mass_2: jax.Array,
            hyperparameters: Mapping[str, Any],
        ) -> jax.Array:
            del hyperparameters
            return (1.0 + tilt * (mass_1 - mass_2)) / normalization

        total_mass, cumulative = _cumulative_mass_moment_grid(
            {},
            tilted_prior,
            minimum_component_mass=MASS_MIN,
            maximum_component_mass=MASS_MAX,
            mass_ratio_quadrature_order=32,
            n_interp_grid=257,
        )
        return jnp.sum(jnp.interp(queries, total_mass, cumulative))

    tilt = 0.02
    step = 1e-5
    actual = float(jax.grad(evaluate)(jnp.asarray(tilt)))
    expected = float(
        (evaluate(jnp.asarray(tilt + step)) - evaluate(jnp.asarray(tilt - step)))
        / (2.0 * step)
    )

    assert actual == pytest.approx(expected, rel=2e-7)


def test_uniform_ordered_joint_density_is_normalized() -> None:
    order = 32
    mass_1, mass_1_weights = _mapped_legendre(order, MASS_MIN, MASS_MAX)
    _, mass_2_weights = _mapped_legendre(order, MASS_MIN, mass_1)
    density = 2.0 / MASS_RANGE**2

    normalization = np.sum(mass_1_weights[:, None] * mass_2_weights * density)

    assert normalization == pytest.approx(1.0, rel=2e-15)


def test_matches_independent_component_mass_quadrature_without_cutoff() -> None:
    frequencies = np.array([10.0, 25.0, 80.0])

    actual = np.asarray(_analytic(jnp.asarray(frequencies)))
    expected = _direct_uniform_spectral_density(frequencies, alpha=math.inf, order=96)

    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert actual.dtype == np.float64


def test_precomputed_mass_moments_are_interpolated_cumulative_queries() -> None:
    frequencies = jnp.array([20.0, 40.0, 80.0])
    actual = precompute_cumulative_mass_moments(
        frequencies,
        {},
        uniform_joint_mass,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        minimum_component_mass=MASS_MIN,
        maximum_component_mass=MASS_MAX,
        alpha=ISCO_ALPHA,
        mass_ratio_quadrature_order=24,
        n_interp_grid=257,
        redshift_quadrature_order=12,
    )
    total_mass, cumulative = _cumulative_mass_moment_grid(
        {},
        uniform_joint_mass,
        minimum_component_mass=MASS_MIN,
        maximum_component_mass=MASS_MAX,
        mass_ratio_quadrature_order=24,
        n_interp_grid=257,
    )
    redshift, _ = _mapped_legendre(12, Z_MIN, Z_MAX)
    upper = ISCO_ALPHA / (
        np.asarray(frequencies)[:, None]
        * (1.0 + redshift[None, :])
        * SOLAR_MASS_IN_SECONDS
    )
    expected = jnp.interp(jnp.asarray(upper), total_mass, cumulative)

    assert actual.shape == (3, 12)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_split_spectrum_matches_combined_wrapper_and_reuses_mass_values() -> None:
    calls: list[tuple[int, ...]] = []

    def counted_prior(
        mass_1: jax.Array,
        mass_2: jax.Array,
        hyperparameters: Mapping[str, Any],
    ) -> jax.Array:
        del mass_2, hyperparameters
        calls.append(mass_1.shape)
        return jnp.full_like(mass_1, 2.0 / MASS_RANGE**2)

    frequencies = jnp.array([20.0, 40.0, 80.0])
    mass_moments = precompute_cumulative_mass_moments(
        frequencies,
        {},
        counted_prior,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        minimum_component_mass=MASS_MIN,
        maximum_component_mass=MASS_MAX,
        alpha=ISCO_ALPHA,
        mass_ratio_quadrature_order=24,
        n_interp_grid=257,
        redshift_quadrature_order=24,
    )
    split = analytic_spectral_density_from_mass_moments(
        frequencies,
        HYPERPARAMETERS,
        constant_rate,
        mass_moments,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        redshift_quadrature_order=24,
    )
    changed_cosmology = {**HYPERPARAMETERS, "H0": 67.0, "Omega_m": 0.3}
    changed = analytic_spectral_density_from_mass_moments(
        frequencies,
        changed_cosmology,
        constant_rate,
        mass_moments,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        redshift_quadrature_order=24,
    )
    combined = analytic_spectral_density(
        frequencies,
        HYPERPARAMETERS,
        constant_rate,
        uniform_joint_mass,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        minimum_component_mass=MASS_MIN,
        maximum_component_mass=MASS_MAX,
        alpha=ISCO_ALPHA,
        quadrature_order=24,
        n_interp_grid=257,
    )

    np.testing.assert_allclose(np.asarray(split), np.asarray(combined), rtol=2e-15)
    assert np.all(np.isfinite(np.asarray(changed)))
    assert calls == [(257, 24)]


def test_split_spectrum_cosmology_gradients_match_combined_wrapper() -> None:
    frequencies = jnp.array([20.0, 40.0])
    mass_moments = precompute_cumulative_mass_moments(
        frequencies,
        {},
        uniform_joint_mass,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        minimum_component_mass=MASS_MIN,
        maximum_component_mass=MASS_MAX,
        alpha=ISCO_ALPHA,
        mass_ratio_quadrature_order=16,
        n_interp_grid=257,
        redshift_quadrature_order=16,
    )

    def split(parameters: jax.Array) -> jax.Array:
        hyperparameters = {
            "H0": parameters[0],
            "Omega_m": parameters[1],
            "rate": parameters[2],
        }
        return jnp.sum(
            analytic_spectral_density_from_mass_moments(
                frequencies,
                hyperparameters,
                constant_rate,
                mass_moments,
                minimum_redshift=Z_MIN,
                maximum_redshift=Z_MAX,
                redshift_quadrature_order=16,
            )
        )

    def combined(parameters: jax.Array) -> jax.Array:
        hyperparameters = {
            "H0": parameters[0],
            "Omega_m": parameters[1],
            "rate": parameters[2],
        }
        return jnp.sum(
            analytic_spectral_density(
                frequencies,
                hyperparameters,
                constant_rate,
                uniform_joint_mass,
                minimum_redshift=Z_MIN,
                maximum_redshift=Z_MAX,
                minimum_component_mass=MASS_MIN,
                maximum_component_mass=MASS_MAX,
                alpha=ISCO_ALPHA,
                quadrature_order=16,
                n_interp_grid=257,
            )
        )

    parameters = jnp.array([70.0, 0.3, 25.0])
    np.testing.assert_allclose(
        np.asarray(jax.grad(split)(parameters)),
        np.asarray(jax.grad(combined)(parameters)),
        rtol=2e-13,
    )


def test_uncut_spectral_slopes_and_omega_conversion() -> None:
    frequencies = jnp.array([20.0, 40.0, 80.0])
    strain = _analytic(frequencies)
    omega = omega_gw_from_spectral_density(
        strain, frequencies, hubble_constant=HYPERPARAMETERS["H0"]
    )

    np.testing.assert_allclose(
        np.asarray(strain[1:] / strain[:-1]),
        np.full(2, 2.0 ** (-7.0 / 3.0)),
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        np.asarray(omega[1:] / omega[:-1]),
        np.full(2, 2.0 ** (2.0 / 3.0)),
        rtol=2e-13,
    )


def test_cutoff_is_an_integration_bound_and_zero_above_support() -> None:
    maximum_observer_frequency = ISCO_ALPHA / (
        2.0 * MASS_MIN * (1.0 + Z_MIN) * SOLAR_MASS_IN_SECONDS
    )
    frequencies = jnp.array(
        [0.5 * maximum_observer_frequency, 1.01 * maximum_observer_frequency]
    )

    strain = np.asarray(_analytic(frequencies, alpha=ISCO_ALPHA))

    assert strain[0] > 0.0
    assert strain[1] == 0.0


def test_matches_independent_component_mass_quadrature_with_cutoff() -> None:
    frequencies = np.array([80.0, 150.0])

    actual = np.asarray(
        _analytic(jnp.asarray(frequencies), alpha=ISCO_ALPHA, quadrature_order=128)
    )
    expected = _direct_uniform_spectral_density(
        frequencies, alpha=ISCO_ALPHA, order=128
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_default_alpha_is_the_shared_dimensionless_isco_constant() -> None:
    """Every entry point must default to the one shared ``ISCO_ALPHA``.

    This module used to export an ``ISCO_ALPHA`` of its own in Hz solar-mass,
    a factor ``1 / SOLAR_MASS_IN_SECONDS`` away from the dimensionless
    constant of the same name in :mod:`astrogwb.waveform`. Identity, not
    equality, is what pins the two together.
    """
    for function in (
        uniform_prior_mass_moments,
        precompute_cumulative_mass_moments,
        analytic_spectral_density,
    ):
        default = inspect.signature(function).parameters["alpha"].default
        assert default is ISCO_ALPHA


def test_is_jittable_with_traced_hyperparameters() -> None:
    def evaluate(frequencies: jax.Array, h0: jax.Array) -> jax.Array:
        hyperparameters = {**HYPERPARAMETERS, "H0": h0}
        return _analytic(frequencies, hyperparameters, quadrature_order=16)

    frequencies = jnp.array([20.0, 40.0])
    expected = _analytic(frequencies, quadrature_order=16)
    actual = jax.jit(evaluate)(frequencies, jnp.asarray(HYPERPARAMETERS["H0"]))

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-13)


def test_order_doubling_converges_for_smooth_nonseparable_population() -> None:
    hyperparameters = {
        "H0": 67.74,
        "Omega_m": 0.31,
        "rate": 40.0,
        "gamma": 1.4,
        "kappa": 4.6,
        "z_peak": 1.8,
        "mass_tilt": 0.02,
    }

    def merger_rate(redshift: jax.Array, parameters: Mapping[str, Any]) -> jax.Array:
        one_plus_z = 1.0 + redshift
        exponent = parameters["gamma"] + parameters["kappa"]
        normalization = 1.0 + (1.0 + parameters["z_peak"]) ** (-exponent)
        return (
            parameters["rate"]
            * normalization
            * one_plus_z ** parameters["gamma"]
            / (1.0 + (one_plus_z / (1.0 + parameters["z_peak"])) ** exponent)
        )

    def joint_mass_prior(
        mass_1: jax.Array,
        mass_2: jax.Array,
        parameters: Mapping[str, Any],
    ) -> jax.Array:
        tilt = parameters["mass_tilt"]
        normalization = 0.5 * MASS_RANGE**2 + tilt * MASS_RANGE**3 / 6.0
        return (1.0 + tilt * (mass_1 - mass_2)) / normalization

    def evaluate(order: int) -> jax.Array:
        return analytic_spectral_density(
            jnp.array([20.0, 30.0]),
            hyperparameters,
            merger_rate,
            joint_mass_prior,
            minimum_redshift=0.0,
            maximum_redshift=4.0,
            minimum_component_mass=MASS_MIN,
            maximum_component_mass=MASS_MAX,
            quadrature_order=order,
        )

    np.testing.assert_allclose(
        np.asarray(evaluate(64)),
        np.asarray(evaluate(128)),
        rtol=1e-5,
    )


def test_uniform_closed_form_matches_direct_component_mass_integral() -> None:
    cutoffs = np.array(
        [
            2.0 * MASS_MIN - 1.0,
            2.0 * MASS_MIN,
            37.0,
            MASS_MIN + MASS_MAX,
            73.0,
            2.0 * MASS_MAX,
            np.inf,
        ]
    )

    actual = np.asarray(
        _uniform_cumulative_mass_moment(
            jnp.asarray(cutoffs),
            minimum_component_mass=MASS_MIN,
            maximum_component_mass=MASS_MAX,
        )
    )
    expected = np.asarray(
        [_direct_uniform_mass_moment(cutoff, order=128) for cutoff in cutoffs]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)
    assert actual[0] == 0.0
    assert actual[1] == 0.0
    assert actual[-1] == actual[-2]


def test_uniform_prior_mass_moments_match_numerical_precompute() -> None:
    frequencies = jnp.array([20.0, 40.0, 80.0])
    kwargs = {
        "minimum_redshift": Z_MIN,
        "maximum_redshift": Z_MAX,
        "minimum_component_mass": MASS_MIN,
        "maximum_component_mass": MASS_MAX,
        "alpha": ISCO_ALPHA,
        "redshift_quadrature_order": 24,
    }

    actual = uniform_prior_mass_moments(frequencies, **kwargs)
    expected = precompute_cumulative_mass_moments(
        frequencies,
        {},
        uniform_joint_mass,
        mass_ratio_quadrature_order=64,
        n_interp_grid=4097,
        **kwargs,
    )

    assert actual.shape == (3, 24)
    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5)


def test_uniform_spectrum_matches_generic_pipeline() -> None:
    frequencies = jnp.array([20.0, 40.0, 80.0])
    mass_moments = uniform_prior_mass_moments(
        frequencies,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        minimum_component_mass=MASS_MIN,
        maximum_component_mass=MASS_MAX,
        alpha=ISCO_ALPHA,
        redshift_quadrature_order=64,
    )

    actual = analytic_spectral_density_from_mass_moments(
        frequencies,
        HYPERPARAMETERS,
        constant_rate,
        mass_moments,
        minimum_redshift=Z_MIN,
        maximum_redshift=Z_MAX,
        redshift_quadrature_order=64,
    )
    expected = _analytic(frequencies, alpha=ISCO_ALPHA)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5)


def test_uniform_prior_mass_moments_is_jittable_and_differentiable() -> None:
    above_support = (
        1.01 * ISCO_ALPHA / (2.0 * MASS_MIN * (1.0 + Z_MIN) * SOLAR_MASS_IN_SECONDS)
    )
    frequencies = jnp.array([20.0, 80.0, above_support])

    def evaluate(values: jax.Array) -> jax.Array:
        return uniform_prior_mass_moments(
            values,
            minimum_redshift=Z_MIN,
            maximum_redshift=Z_MAX,
            minimum_component_mass=MASS_MIN,
            maximum_component_mass=MASS_MAX,
            alpha=ISCO_ALPHA,
            redshift_quadrature_order=16,
        )

    jitted = np.asarray(jax.jit(evaluate)(frequencies))
    gradient = np.asarray(
        jax.grad(lambda values: jnp.sum(evaluate(values)))(frequencies)
    )

    np.testing.assert_allclose(jitted, np.asarray(evaluate(frequencies)), rtol=1e-12)
    assert np.all(jitted[-1] == 0.0)
    assert np.all(np.isfinite(gradient))
    assert gradient[-1] == 0.0
