from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.cosmology import MPC_IN_METERS, SPEED_OF_LIGHT, hubble_constant_si
from astrogwb.gwb import (
    ISCO_ALPHA,
    analytic_spectral_density,
    omega_gw_from_spectral_density,
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
Q_MIN = 0.2


def constant_rate(redshift: jax.Array, hyperparameters: Mapping[str, Any]) -> jax.Array:
    return jnp.full_like(redshift, hyperparameters["rate"])


def uniform_total_mass(
    total_mass: jax.Array, hyperparameters: Mapping[str, Any]
) -> jax.Array:
    del hyperparameters
    return jnp.full_like(total_mass, 1.0 / (MASS_MAX - MASS_MIN))


def uniform_mass_ratio(
    mass_ratio: jax.Array, hyperparameters: Mapping[str, Any]
) -> jax.Array:
    del hyperparameters
    return jnp.full_like(mass_ratio, 1.0 / (1.0 - Q_MIN))


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
        uniform_total_mass,
        uniform_mass_ratio,
        z_min=Z_MIN,
        z_max=Z_MAX,
        total_mass_min=MASS_MIN,
        total_mass_max=MASS_MAX,
        q_min=Q_MIN,
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


def _uniform_closed_form(frequencies: np.ndarray) -> np.ndarray:
    mass_moment = (
        3.0
        / 8.0
        * (MASS_MAX ** (8.0 / 3.0) - MASS_MIN ** (8.0 / 3.0))
        / (MASS_MAX - MASS_MIN)
    )
    antiderivative = lambda q: np.log1p(q) + 1.0 / (1.0 + q)
    mass_ratio_moment = (antiderivative(1.0) - antiderivative(Q_MIN)) / (1.0 - Q_MIN)
    redshift_moment = (
        3.0
        * HYPERPARAMETERS["rate"]
        * ((1.0 + Z_MIN) ** (-1.0 / 3.0) - (1.0 + Z_MAX) ** (-1.0 / 3.0))
    )
    return (
        _strain_coefficient(HYPERPARAMETERS["H0"])
        * frequencies ** (-7.0 / 3.0)
        * mass_moment
        * mass_ratio_moment
        * redshift_moment
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


def test_matches_closed_form_without_cutoff() -> None:
    frequencies = np.array([10.0, 25.0, 80.0])

    actual = np.asarray(_analytic(jnp.asarray(frequencies)))
    expected = _uniform_closed_form(frequencies)

    np.testing.assert_allclose(actual, expected, rtol=2e-13)
    assert actual.dtype == np.float64


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
    maximum_observer_frequency = ISCO_ALPHA / (MASS_MIN * (1.0 + Z_MIN))
    frequencies = jnp.array(
        [0.5 * maximum_observer_frequency, 1.01 * maximum_observer_frequency]
    )

    strain = np.asarray(_analytic(frequencies, alpha=ISCO_ALPHA))

    assert strain[0] > 0.0
    assert strain[1] == 0.0


def test_matches_independent_direct_three_dimensional_quadrature() -> None:
    frequency = 150.0
    order = 96
    z, wz = _mapped_legendre(order, Z_MIN, Z_MAX)
    q, wq = _mapped_legendre(order, Q_MIN, 1.0)
    mass_upper = np.minimum(MASS_MAX, ISCO_ALPHA / (frequency * (1.0 + z)))
    mass, wm = _mapped_legendre(order, MASS_MIN, mass_upper)

    integrand = (
        HYPERPARAMETERS["rate"]
        / (1.0 + z[:, None, None]) ** (4.0 / 3.0)
        * mass[:, :, None] ** (5.0 / 3.0)
        / (MASS_MAX - MASS_MIN)
        * q[None, None, :]
        / (1.0 + q[None, None, :]) ** 2
        / (1.0 - Q_MIN)
    )
    population_moment = np.sum(
        wz[:, None, None] * wm[:, :, None] * wq[None, None, :] * integrand
    )
    expected = (
        _strain_coefficient(HYPERPARAMETERS["H0"])
        * frequency ** (-7.0 / 3.0)
        * population_moment
    )

    actual = float(_analytic(jnp.array([frequency]), alpha=ISCO_ALPHA)[0])

    assert actual == pytest.approx(expected, rel=2e-13)


def test_default_isco_alpha_uses_astrophysical_units() -> None:
    expected = SPEED_OF_LIGHT**3 / (
        6.0 ** (3.0 / 2.0) * np.pi * GRAVITATIONAL_CONSTANT_SI * SOLAR_MASS_KG
    )

    assert ISCO_ALPHA == pytest.approx(expected, rel=1e-15)
    assert 4390.0 < ISCO_ALPHA < 4400.0


def test_is_jittable_with_traced_hyperparameters() -> None:
    def evaluate(frequencies: jax.Array, h0: jax.Array) -> jax.Array:
        hyperparameters = {**HYPERPARAMETERS, "H0": h0}
        return _analytic(frequencies, hyperparameters)

    frequencies = jnp.array([20.0, 40.0])
    expected = _analytic(frequencies)
    actual = jax.jit(evaluate)(frequencies, jnp.asarray(HYPERPARAMETERS["H0"]))

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-13)


def test_order_doubling_converges_for_smooth_population() -> None:
    hyperparameters = {
        "H0": 67.74,
        "Omega_m": 0.31,
        "rate": 40.0,
        "gamma": 1.4,
        "kappa": 4.6,
        "z_peak": 1.8,
        "mass_mean": 30.0,
        "mass_sigma": 8.0,
        "mass_normalization": 19.84286146940555,
        "q_beta": 1.3,
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

    def mass_prior(total_mass: jax.Array, parameters: Mapping[str, Any]) -> jax.Array:
        standardized = (total_mass - parameters["mass_mean"]) / parameters["mass_sigma"]
        return jnp.exp(-0.5 * standardized**2) / parameters["mass_normalization"]

    def q_prior(mass_ratio: jax.Array, parameters: Mapping[str, Any]) -> jax.Array:
        exponent = parameters["q_beta"] + 1.0
        return exponent * mass_ratio ** parameters["q_beta"] / (1.0 - Q_MIN**exponent)

    def evaluate(order: int) -> jax.Array:
        return analytic_spectral_density(
            jnp.array([100.0, 120.0]),
            hyperparameters,
            merger_rate,
            mass_prior,
            q_prior,
            z_min=0.0,
            z_max=4.0,
            total_mass_min=MASS_MIN,
            total_mass_max=MASS_MAX,
            q_min=Q_MIN,
            quadrature_order=order,
        )

    np.testing.assert_allclose(
        np.asarray(evaluate(64)),
        np.asarray(evaluate(128)),
        rtol=1e-6,
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
    ("overrides", "message"),
    [
        ({"z_min": -1.0}, "redshift bounds"),
        ({"z_max": Z_MIN}, "redshift bounds"),
        ({"total_mass_min": 0.0}, "total-mass bounds"),
        ({"total_mass_max": MASS_MIN}, "total-mass bounds"),
        ({"q_min": -0.1}, "q_min"),
        ({"q_min": 1.0}, "q_min"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": np.nan}, "alpha"),
        ({"quadrature_order": 0}, "quadrature_order"),
        ({"quadrature_order": 1.5}, "quadrature_order"),
        ({"quadrature_order": True}, "quadrature_order"),
    ],
)
def test_rejects_invalid_static_inputs(overrides: dict[str, Any], message: str) -> None:
    arguments = {
        "z_min": Z_MIN,
        "z_max": Z_MAX,
        "total_mass_min": MASS_MIN,
        "total_mass_max": MASS_MAX,
        "q_min": Q_MIN,
        "alpha": math.inf,
        "quadrature_order": 16,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        analytic_spectral_density(
            jnp.array([20.0]),
            HYPERPARAMETERS,
            constant_rate,
            uniform_total_mass,
            uniform_mass_ratio,
            **arguments,
        )


def test_requires_cosmological_hyperparameters() -> None:
    with pytest.raises(KeyError, match="H0"):
        _analytic(jnp.array([20.0]), {"Omega_m": 0.3, "rate": 25.0})
    with pytest.raises(KeyError, match="Omega_m"):
        _analytic(jnp.array([20.0]), {"H0": 70.0, "rate": 25.0})
