from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.cosmology import MPC_IN_METERS, SPEED_OF_LIGHT, hubble_constant_si
from astrogwb.gwb import (
    ISCO_ALPHA,
    analytic_spectral_density,
    analytic_spectral_density_from_mass_moments,
    make_cumulative_mass_moment_fn,
    omega_gw_from_spectral_density,
    precompute_cumulative_mass_moments,
)
from astrogwb.gwb.analytic import (
    _evaluate_piecewise_cumulative,
    _piecewise_legendre_coefficients,
    _piecewise_legendre_scheme,
)
from astrogwb.utils import SECONDS_PER_YEAR
from numpy.polynomial.legendre import leggauss

jax.config.update("jax_enable_x64", True)

GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
SOLAR_MASS_KG = 1.988409870698051e30
GPC_IN_METERS = 1.0e3 * MPC_IN_METERS

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
        z_min=Z_MIN,
        z_max=Z_MAX,
        component_mass_min=MASS_MIN,
        component_mass_max=MASS_MAX,
        alpha=alpha,
        quadrature_order=quadrature_order,
    )


def _strain_coefficient(h0: float) -> float:
    return (
        2.0
        * (GRAVITATIONAL_CONSTANT_SI * SOLAR_MASS_KG) ** (5.0 / 3.0)
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
                _direct_uniform_mass_moment(alpha / (frequency * (1.0 + value)), order)
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


def test_piecewise_legendre_cumulative_is_exact_for_cubic_polynomial() -> None:
    scheme = _piecewise_legendre_scheme(1.0, 3.0, 4, 4)
    nodes = jnp.asarray(scheme.nodes)
    values = nodes**3 - 2.0 * nodes + 1.0
    coefficients, cumulative = _piecewise_legendre_coefficients(values, scheme)
    queries = jnp.array([0.0, 2.0, 2.7, 4.0, 5.3, 6.0, 8.0])

    actual = _evaluate_piecewise_cumulative(coefficients, cumulative, scheme, queries)
    clipped = jnp.clip(queries, 2.0, 6.0)
    antiderivative = 0.25 * clipped**4 - clipped**2 + clipped
    lower_antiderivative = 0.25 * 2.0**4 - 2.0**2 + 2.0
    expected = antiderivative - lower_antiderivative

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-13)
    assert actual[0] == 0.0
    assert actual[-1] == cumulative[-1]


def test_piecewise_legendre_cumulative_is_jittable_and_differentiable() -> None:
    scheme = _piecewise_legendre_scheme(1.0, 3.0, 4, 3)
    nodes = jnp.asarray(scheme.nodes)
    queries = jnp.array([2.5, 3.75, 5.5])

    def evaluate(scale: jax.Array) -> jax.Array:
        coefficients, cumulative = _piecewise_legendre_coefficients(
            scale * nodes**2, scheme
        )
        return jnp.sum(
            _evaluate_piecewise_cumulative(coefficients, cumulative, scheme, queries)
        )

    actual = jax.jit(evaluate)(jnp.asarray(2.0))
    derivative = jax.grad(evaluate)(jnp.asarray(2.0))
    expected_derivative = jnp.sum((queries**3 - 2.0**3) / 3.0)

    assert actual == pytest.approx(2.0 * float(expected_derivative), rel=2e-13)
    assert derivative == pytest.approx(float(expected_derivative), rel=2e-13)


def test_cumulative_mass_moment_matches_direct_component_mass_integral() -> None:
    cumulative_mass_moment_fn = make_cumulative_mass_moment_fn(
        {},
        uniform_joint_mass,
        component_mass_min=MASS_MIN,
        component_mass_max=MASS_MAX,
        mass_ratio_quadrature_order=64,
        total_mass_interval_count=32,
        total_mass_legendre_order=16,
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

    actual = np.asarray(cumulative_mass_moment_fn(jnp.asarray(cutoffs)))
    expected = np.asarray(
        [_direct_uniform_mass_moment(cutoff, order=128) for cutoff in cutoffs]
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
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

    cumulative_mass_moment_fn = make_cumulative_mass_moment_fn(
        {},
        guarded_prior,
        component_mass_min=MASS_MIN,
        component_mass_max=MASS_MAX,
        mass_ratio_quadrature_order=12,
        total_mass_interval_count=8,
        total_mass_legendre_order=6,
    )
    values = cumulative_mass_moment_fn(jnp.linspace(0.0, 100.0, 17))

    assert calls == [(8, 6, 12)]
    assert np.all(np.isfinite(np.asarray(values)))


def test_featured_mass_population_converges_when_doubling_intervals() -> None:
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

    def evaluate(interval_count: int) -> jax.Array:
        cumulative = make_cumulative_mass_moment_fn(
            {},
            featured_prior,
            component_mass_min=MASS_MIN,
            component_mass_max=MASS_MAX,
            mass_ratio_quadrature_order=64,
            total_mass_interval_count=interval_count,
            total_mass_legendre_order=16,
        )
        return cumulative(jnp.linspace(2.0 * MASS_MIN, 2.0 * MASS_MAX, 101))

    np.testing.assert_allclose(
        np.asarray(evaluate(16)),
        np.asarray(evaluate(32)),
        rtol=1e-10,
        atol=1e-11,
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

        cumulative = make_cumulative_mass_moment_fn(
            {},
            tilted_prior,
            component_mass_min=MASS_MIN,
            component_mass_max=MASS_MAX,
            mass_ratio_quadrature_order=32,
            total_mass_interval_count=16,
            total_mass_legendre_order=12,
        )
        return jnp.sum(cumulative(queries))

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

    np.testing.assert_allclose(actual, expected, rtol=2e-12)
    assert actual.dtype == np.float64


def test_precomputed_mass_moments_are_exact_cumulative_queries() -> None:
    cumulative = make_cumulative_mass_moment_fn(
        {},
        uniform_joint_mass,
        component_mass_min=MASS_MIN,
        component_mass_max=MASS_MAX,
        mass_ratio_quadrature_order=24,
        total_mass_interval_count=16,
        total_mass_legendre_order=10,
    )
    frequencies = jnp.array([20.0, 40.0, 80.0])
    actual = precompute_cumulative_mass_moments(
        frequencies,
        cumulative,
        z_min=Z_MIN,
        z_max=Z_MAX,
        alpha=ISCO_ALPHA,
        redshift_quadrature_order=12,
    )
    redshift, _ = _mapped_legendre(12, Z_MIN, Z_MAX)
    upper = ISCO_ALPHA / (np.asarray(frequencies)[:, None] * (1.0 + redshift[None, :]))
    expected = cumulative(jnp.asarray(upper))

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
    cumulative = make_cumulative_mass_moment_fn(
        {},
        counted_prior,
        component_mass_min=MASS_MIN,
        component_mass_max=MASS_MAX,
        mass_ratio_quadrature_order=24,
        total_mass_interval_count=16,
        total_mass_legendre_order=10,
    )
    mass_moments = precompute_cumulative_mass_moments(
        frequencies,
        cumulative,
        z_min=Z_MIN,
        z_max=Z_MAX,
        alpha=ISCO_ALPHA,
        redshift_quadrature_order=24,
    )
    split = analytic_spectral_density_from_mass_moments(
        frequencies,
        HYPERPARAMETERS,
        constant_rate,
        mass_moments,
        z_min=Z_MIN,
        z_max=Z_MAX,
        redshift_quadrature_order=24,
    )
    changed_cosmology = {**HYPERPARAMETERS, "H0": 67.0, "Omega_m": 0.3}
    changed = analytic_spectral_density_from_mass_moments(
        frequencies,
        changed_cosmology,
        constant_rate,
        mass_moments,
        z_min=Z_MIN,
        z_max=Z_MAX,
        redshift_quadrature_order=24,
    )
    combined = analytic_spectral_density(
        frequencies,
        HYPERPARAMETERS,
        constant_rate,
        uniform_joint_mass,
        z_min=Z_MIN,
        z_max=Z_MAX,
        component_mass_min=MASS_MIN,
        component_mass_max=MASS_MAX,
        alpha=ISCO_ALPHA,
        quadrature_order=24,
        total_mass_interval_count=16,
        total_mass_legendre_order=10,
    )

    np.testing.assert_allclose(np.asarray(split), np.asarray(combined), rtol=2e-15)
    assert np.all(np.isfinite(np.asarray(changed)))
    assert calls == [(16, 10, 24)]


def test_split_spectrum_cosmology_gradients_match_combined_wrapper() -> None:
    frequencies = jnp.array([20.0, 40.0])
    cumulative = make_cumulative_mass_moment_fn(
        {},
        uniform_joint_mass,
        component_mass_min=MASS_MIN,
        component_mass_max=MASS_MAX,
        mass_ratio_quadrature_order=16,
        total_mass_interval_count=8,
        total_mass_legendre_order=8,
    )
    mass_moments = precompute_cumulative_mass_moments(
        frequencies,
        cumulative,
        z_min=Z_MIN,
        z_max=Z_MAX,
        alpha=ISCO_ALPHA,
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
                z_min=Z_MIN,
                z_max=Z_MAX,
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
                z_min=Z_MIN,
                z_max=Z_MAX,
                component_mass_min=MASS_MIN,
                component_mass_max=MASS_MAX,
                alpha=ISCO_ALPHA,
                quadrature_order=16,
                total_mass_interval_count=8,
                total_mass_legendre_order=8,
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
    maximum_observer_frequency = ISCO_ALPHA / (2.0 * MASS_MIN * (1.0 + Z_MIN))
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

    np.testing.assert_allclose(actual, expected, rtol=5e-6)


def test_default_isco_alpha_uses_astrophysical_units() -> None:
    expected = SPEED_OF_LIGHT**3 / (
        6.0 ** (3.0 / 2.0) * np.pi * GRAVITATIONAL_CONSTANT_SI * SOLAR_MASS_KG
    )

    assert ISCO_ALPHA == pytest.approx(expected, rel=1e-15)
    assert 4390.0 < ISCO_ALPHA < 4400.0


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
            z_min=0.0,
            z_max=4.0,
            component_mass_min=MASS_MIN,
            component_mass_max=MASS_MAX,
            quadrature_order=order,
        )

    np.testing.assert_allclose(
        np.asarray(evaluate(64)),
        np.asarray(evaluate(128)),
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "frequencies",
    [
        jnp.array([]),
        jnp.array([0.0]),
        jnp.array([-1.0]),
        jnp.array([jnp.inf]),
        jnp.array([jnp.nan]),
        jnp.ones((2, 2)),
    ],
)
def test_rejects_invalid_frequencies(frequencies: jax.Array) -> None:
    with pytest.raises(ValueError, match="frequencies"):
        _analytic(frequencies)


@pytest.mark.parametrize(
    "mass_moments",
    [
        jnp.ones(4),
        jnp.ones((2, 3)),
        jnp.ones((2, 4, 1)),
    ],
)
def test_split_spectrum_rejects_incompatible_mass_moment_shape(
    mass_moments: jax.Array,
) -> None:
    with pytest.raises(ValueError, match="cumulative_mass_moments must have shape"):
        analytic_spectral_density_from_mass_moments(
            jnp.array([20.0, 40.0]),
            HYPERPARAMETERS,
            constant_rate,
            mass_moments,
            z_min=Z_MIN,
            z_max=Z_MAX,
            redshift_quadrature_order=4,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"z_min": -1.0}, "redshift bounds"),
        ({"z_max": Z_MIN}, "redshift bounds"),
        ({"component_mass_min": 0.0}, "component-mass bounds"),
        ({"component_mass_max": MASS_MIN}, "component-mass bounds"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": np.nan}, "alpha"),
        ({"quadrature_order": 0}, "quadrature_order"),
        ({"quadrature_order": 1.5}, "quadrature_order"),
        ({"quadrature_order": True}, "quadrature_order"),
        ({"total_mass_interval_count": 1}, "total_mass_interval_count"),
        ({"total_mass_interval_count": 3}, "total_mass_interval_count"),
        ({"total_mass_interval_count": True}, "total_mass_interval_count"),
        ({"total_mass_legendre_order": 0}, "total_mass_legendre_order"),
        ({"total_mass_legendre_order": 1.5}, "total_mass_legendre_order"),
        ({"total_mass_legendre_order": True}, "total_mass_legendre_order"),
    ],
)
def test_rejects_invalid_static_inputs(overrides: dict[str, Any], message: str) -> None:
    arguments: dict[str, Any] = {
        "z_min": Z_MIN,
        "z_max": Z_MAX,
        "component_mass_min": MASS_MIN,
        "component_mass_max": MASS_MAX,
        "alpha": math.inf,
        "quadrature_order": 16,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        analytic_spectral_density(
            jnp.array([20.0]),
            HYPERPARAMETERS,
            constant_rate,
            uniform_joint_mass,
            **arguments,
        )


def test_requires_cosmological_hyperparameters() -> None:
    with pytest.raises(KeyError, match="H0"):
        _analytic(jnp.array([20.0]), {"Omega_m": 0.3, "rate": 25.0})
    with pytest.raises(KeyError, match="Omega_m"):
        _analytic(jnp.array([20.0]), {"H0": 70.0, "rate": 25.0})
