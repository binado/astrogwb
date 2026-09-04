from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.distributions import TruncatedNormal

from astrogwb.constants import ISCO_ALPHA, SOLAR_MASS_IN_SECONDS
from astrogwb.gwb import (
    JointMassFunction,
    PopulationFunction,
    analytic_spectral_density,
    analytic_spectral_density_from_mass_moments,
    precompute_cumulative_mass_moments,
    uniform_prior_mass_moments,
)


class IntegrationBounds(TypedDict):
    minimum_redshift: float
    maximum_redshift: float
    minimum_component_mass: float
    maximum_component_mass: float


type AnalyticFunction = Callable[[jax.Array, Mapping[str, Any]], jax.Array]


@pytest.fixture
def frequencies() -> jax.Array:
    return jnp.geomspace(5.0, 300.0, 48)


@pytest.fixture
def transform_frequencies() -> jax.Array:
    # Cover full and partial mass support without near-boundary cancellation
    # dominating relative eager/JIT comparisons of tiny uniform moments.
    return jnp.asarray([5.0, 20.0, 80.0])


@pytest.fixture
def hyperparameters() -> dict[str, float]:
    return {
        "H0": 67.74,
        "Omega_m": 0.31,
        "rate": 40.0,
        "gamma": 1.4,
        "mass_peak": 30.0,
        "mass_sigma": 5.0,
    }


@pytest.fixture
def bounds() -> IntegrationBounds:
    return {
        "minimum_redshift": 0.0,
        "maximum_redshift": 4.0,
        "minimum_component_mass": 5.0,
        "maximum_component_mass": 45.0,
    }


@pytest.fixture
def alpha() -> float:
    return ISCO_ALPHA


@pytest.fixture
def quadrature_orders() -> tuple[int, int, int]:
    return 16, 32, 64


@pytest.fixture
def interpolation_grid_sizes() -> tuple[int, int, int]:
    return 1025, 2049, 4097


@pytest.fixture
def fast_resolution(quadrature_orders: tuple[int, int, int]) -> dict[str, int]:
    return {"quadrature_order": quadrature_orders[0], "n_interp_grid": 257}


@pytest.fixture
def merger_rate_fn() -> PopulationFunction:
    def merger_rate(redshift: jax.Array, parameters: Mapping[str, Any]) -> jax.Array:
        return parameters["rate"] * (1.0 + redshift) ** parameters["gamma"]

    return merger_rate


@pytest.fixture
def joint_mass_priors(bounds: IntegrationBounds) -> dict[str, JointMassFunction]:
    minimum_mass = bounds["minimum_component_mass"]
    maximum_mass = bounds["maximum_component_mass"]
    uniform_density = 2.0 / (maximum_mass - minimum_mass) ** 2

    def uniform(
        mass_1: jax.Array, mass_2: jax.Array, parameters: Mapping[str, Any]
    ) -> jax.Array:
        del mass_2, parameters
        return jnp.full_like(mass_1, uniform_density)

    def gaussian(
        mass_1: jax.Array, mass_2: jax.Array, parameters: Mapping[str, Any]
    ) -> jax.Array:
        component_mass = TruncatedNormal(
            parameters["mass_peak"],
            parameters["mass_sigma"],
            low=minimum_mass,
            high=maximum_mass,
        )
        # Identical independent components assign half their probability to
        # each ordering, so conditioning on m1 > m2 multiplies the PDF by two.
        density = 2.0 * jnp.exp(
            component_mass.log_prob(mass_1) + component_mass.log_prob(mass_2)
        )
        in_triangle = (
            (mass_2 >= minimum_mass) & (mass_1 > mass_2) & (mass_1 <= maximum_mass)
        )
        return jnp.where(in_triangle, density, 0.0)

    return {"uniform": uniform, "gaussian": gaussian}


@pytest.fixture
def analytic_functions(
    transform_frequencies: jax.Array,
    hyperparameters: Mapping[str, float],
    bounds: IntegrationBounds,
    alpha: float,
    fast_resolution: Mapping[str, int],
    merger_rate_fn: PopulationFunction,
    joint_mass_priors: Mapping[str, JointMassFunction],
) -> dict[str, AnalyticFunction]:
    """Bind static inputs to public APIs; the contraction reuses fixed moments."""
    order = fast_resolution["quadrature_order"]
    precompute = partial(
        precompute_cumulative_mass_moments,
        joint_mass_prior_fn=joint_mass_priors["gaussian"],
        **bounds,
        alpha=alpha,
        mass_ratio_quadrature_order=order,
        redshift_quadrature_order=order,
        n_interp_grid=fast_resolution["n_interp_grid"],
    )
    mass_moments = precompute(transform_frequencies, hyperparameters)
    contract = partial(
        analytic_spectral_density_from_mass_moments,
        merger_rate_fn=merger_rate_fn,
        cumulative_mass_moments=mass_moments,
        minimum_redshift=bounds["minimum_redshift"],
        maximum_redshift=bounds["maximum_redshift"],
        redshift_quadrature_order=order,
    )
    combined = partial(
        analytic_spectral_density,
        merger_rate_fn=merger_rate_fn,
        joint_mass_prior_fn=joint_mass_priors["gaussian"],
        **bounds,
        alpha=alpha,
        **fast_resolution,
    )
    return {"precompute": precompute, "contract": contract, "combined": combined}


@pytest.fixture(
    params=["numpy_scalar", "jax_scalar", "numpy_singleton", "jax_singleton"]
)
def array_hyperparameters(
    request: pytest.FixtureRequest, hyperparameters: Mapping[str, float]
) -> dict[str, Any]:
    convert = np.asarray if request.param.startswith("numpy") else jnp.asarray
    singleton = request.param.endswith("singleton")
    return {
        name: convert([value] if singleton else value)
        for name, value in hyperparameters.items()
    }


@pytest.fixture(params=["finite", "uncut"])
def equivalence_alpha(request: pytest.FixtureRequest, alpha: float) -> float:
    return alpha if request.param == "finite" else math.inf


@pytest.fixture(params=[0.0, 0.5], ids=["zmin=0", "zmin=0.5"])
def cutoff_bounds(
    request: pytest.FixtureRequest, bounds: IntegrationBounds
) -> IntegrationBounds:
    return {**bounds, "minimum_redshift": request.param}


@pytest.fixture(params=[1.0, 2.0], ids=["alpha=ISCO", "alpha=2*ISCO"])
def cutoff_alpha(request: pytest.FixtureRequest, alpha: float) -> float:
    return request.param * alpha


@pytest.fixture
def cutoff_frequencies(
    cutoff_bounds: IntegrationBounds, cutoff_alpha: float
) -> jax.Array:
    maximum_observer_frequency = cutoff_alpha / (
        2.0
        * cutoff_bounds["minimum_component_mass"]
        * (1.0 + cutoff_bounds["minimum_redshift"])
        * SOLAR_MASS_IN_SECONDS
    )
    return maximum_observer_frequency * jnp.asarray([0.5, 1.0, 1.01, 2.0, 10.0])


def _successive_fractional_rms(values: Sequence[jax.Array]) -> tuple[float, float]:
    """Compare successive refinements using the same finest-spectrum scale."""
    coarse, medium, fine = (np.asarray(value) for value in values)
    assert np.all(np.isfinite(fine))
    assert np.all(fine > 0.0)
    coarse_residual = np.sqrt(np.mean(((coarse - medium) / fine) ** 2))
    fine_residual = np.sqrt(np.mean(((medium - fine) / fine) ** 2))
    return float(coarse_residual), float(fine_residual)


@pytest.mark.parametrize("function_name", ["precompute", "contract", "combined"])
def test_analytic_function_is_jittable(
    function_name: str,
    analytic_functions: Mapping[str, AnalyticFunction],
    transform_frequencies: jax.Array,
    hyperparameters: Mapping[str, float],
) -> None:
    function = analytic_functions[function_name]
    expected = function(transform_frequencies, hyperparameters)
    actual = jax.jit(function)(transform_frequencies, hyperparameters)

    assert np.all(np.isfinite(np.asarray(actual)))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


def test_uniform_mass_moments_are_jittable(
    transform_frequencies: jax.Array,
    bounds: IntegrationBounds,
    alpha: float,
    fast_resolution: Mapping[str, int],
) -> None:
    function = partial(
        uniform_prior_mass_moments,
        **bounds,
        alpha=alpha,
        redshift_quadrature_order=fast_resolution["quadrature_order"],
    )
    actual = jax.jit(function)(transform_frequencies)
    assert np.all(np.isfinite(np.asarray(actual)))
    np.testing.assert_allclose(
        actual, function(transform_frequencies), rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize(
    ("function_name", "parameter_names"),
    [
        ("precompute", ("mass_peak", "mass_sigma")),
        ("contract", ("H0", "Omega_m", "rate", "gamma")),
        ("combined", ("H0", "Omega_m", "rate", "gamma", "mass_peak", "mass_sigma")),
    ],
)
def test_analytic_function_has_finite_hyperparameter_gradients(
    function_name: str,
    parameter_names: tuple[str, ...],
    analytic_functions: Mapping[str, AnalyticFunction],
    transform_frequencies: jax.Array,
    hyperparameters: Mapping[str, float],
) -> None:
    function = analytic_functions[function_name]
    gradient = jax.jit(
        jax.grad(
            lambda parameters: jnp.sum(function(transform_frequencies, parameters))
        )
    )(hyperparameters)

    for name in parameter_names:
        assert np.isfinite(np.asarray(gradient[name])), name


def test_uniform_shortcut_matches_generic_pipeline(
    frequencies: jax.Array,
    hyperparameters: Mapping[str, float],
    bounds: IntegrationBounds,
    equivalence_alpha: float,
    quadrature_orders: tuple[int, int, int],
    interpolation_grid_sizes: tuple[int, int, int],
    merger_rate_fn: PopulationFunction,
    joint_mass_priors: Mapping[str, JointMassFunction],
) -> None:
    order = quadrature_orders[-1]
    mass_moments = uniform_prior_mass_moments(
        frequencies, **bounds, alpha=equivalence_alpha, redshift_quadrature_order=order
    )
    shortcut = analytic_spectral_density_from_mass_moments(
        frequencies,
        hyperparameters,
        merger_rate_fn,
        mass_moments,
        minimum_redshift=bounds["minimum_redshift"],
        maximum_redshift=bounds["maximum_redshift"],
        redshift_quadrature_order=order,
    )
    generic = analytic_spectral_density(
        frequencies,
        hyperparameters,
        merger_rate_fn,
        joint_mass_priors["uniform"],
        **bounds,
        alpha=equivalence_alpha,
        quadrature_order=order,
        n_interp_grid=interpolation_grid_sizes[-1],
    )

    np.testing.assert_allclose(generic, shortcut, rtol=5e-5, atol=0.0)


@pytest.mark.parametrize("function_name", ["precompute", "contract", "combined"])
def test_scalar_hyperparameter_representations_are_equivalent(
    function_name: str,
    analytic_functions: Mapping[str, AnalyticFunction],
    transform_frequencies: jax.Array,
    hyperparameters: Mapping[str, float],
    array_hyperparameters: Mapping[str, Any],
) -> None:
    function = analytic_functions[function_name]
    expected = function(transform_frequencies, hyperparameters)
    actual = function(transform_frequencies, array_hyperparameters)

    assert actual.shape == expected.shape
    assert np.all(np.isfinite(np.asarray(actual)))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("prior_name", ["uniform", "gaussian"])
def test_spectrum_converges_with_quadrature_order(
    prior_name: str,
    frequencies: jax.Array,
    hyperparameters: Mapping[str, float],
    bounds: IntegrationBounds,
    alpha: float,
    quadrature_orders: tuple[int, int, int],
    interpolation_grid_sizes: tuple[int, int, int],
    merger_rate_fn: PopulationFunction,
    joint_mass_priors: Mapping[str, JointMassFunction],
) -> None:
    values = [
        analytic_spectral_density(
            frequencies,
            hyperparameters,
            merger_rate_fn,
            joint_mass_priors[prior_name],
            **bounds,
            alpha=alpha,
            quadrature_order=order,
            n_interp_grid=interpolation_grid_sizes[-1],
        )
        for order in quadrature_orders
    ]
    coarse_residual, fine_residual = _successive_fractional_rms(values)

    # Require substantial improvement, without prescribing a universal
    # Gauss-Legendre convergence exponent for the cutoff spectrum.
    assert coarse_residual > 0.0
    assert fine_residual / coarse_residual < 0.25
    assert fine_residual < 1e-4


@pytest.mark.parametrize("prior_name", ["uniform", "gaussian"])
def test_spectrum_converges_with_interpolation_grid_size(
    prior_name: str,
    frequencies: jax.Array,
    hyperparameters: Mapping[str, float],
    bounds: IntegrationBounds,
    alpha: float,
    quadrature_orders: tuple[int, int, int],
    interpolation_grid_sizes: tuple[int, int, int],
    merger_rate_fn: PopulationFunction,
    joint_mass_priors: Mapping[str, JointMassFunction],
) -> None:
    values = [
        analytic_spectral_density(
            frequencies,
            hyperparameters,
            merger_rate_fn,
            joint_mass_priors[prior_name],
            **bounds,
            alpha=alpha,
            quadrature_order=quadrature_orders[-1],
            n_interp_grid=size,
        )
        for size in interpolation_grid_sizes
    ]
    coarse_residual, fine_residual = _successive_fractional_rms(values)

    # Doubling N - 1 halves the spacing; second-order convergence gives
    # roughly a quarter-sized residual, with room for interpolation phase.
    assert coarse_residual > 0.0
    assert 0.15 < fine_residual / coarse_residual < 0.35


@pytest.mark.parametrize("branch", ["uniform_shortcut", "uniform", "gaussian"])
def test_spectrum_is_zero_at_and_above_maximum_observer_frequency(
    branch: str,
    cutoff_frequencies: jax.Array,
    hyperparameters: Mapping[str, float],
    cutoff_bounds: IntegrationBounds,
    cutoff_alpha: float,
    fast_resolution: Mapping[str, int],
    merger_rate_fn: PopulationFunction,
    joint_mass_priors: Mapping[str, JointMassFunction],
) -> None:
    order = fast_resolution["quadrature_order"]
    if branch == "uniform_shortcut":
        mass_moments = uniform_prior_mass_moments(
            cutoff_frequencies,
            **cutoff_bounds,
            alpha=cutoff_alpha,
            redshift_quadrature_order=order,
        )
        spectrum = analytic_spectral_density_from_mass_moments(
            cutoff_frequencies,
            hyperparameters,
            merger_rate_fn,
            mass_moments,
            minimum_redshift=cutoff_bounds["minimum_redshift"],
            maximum_redshift=cutoff_bounds["maximum_redshift"],
            redshift_quadrature_order=order,
        )
    else:
        spectrum = analytic_spectral_density(
            cutoff_frequencies,
            hyperparameters,
            merger_rate_fn,
            joint_mass_priors[branch],
            **cutoff_bounds,
            alpha=cutoff_alpha,
            **fast_resolution,
        )

    assert spectrum[0] > 0.0
    np.testing.assert_array_equal(np.asarray(spectrum[1:]), 0.0)
