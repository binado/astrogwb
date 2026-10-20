"""The compound-Poisson amplitude density and the likelihood built on it.

The density tests use toy point sets whose amplitudes lie on a lattice, where
the exact density is a Poisson-weighted mixture of Gaussians summed directly,
plus a Monte Carlo draw of an off-lattice toy. The likelihood tests use the
closed-form mock reference catalog of ``astrogwb_mock_population``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    INTRINSIC_DENSITY_SITES,
    NUM_REDSHIFT_NODES,
    build_reference_catalog,
    build_reference_spectrum,
    mock_population,
)
from jax.typing import ArrayLike
from scipy import stats

from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.inference import (
    AmplitudeDensityMethod,
    ImportanceCompoundPoissonLikelihood,
    ImportanceGaussianLikelihood,
    LogDensityFn,
    Network,
    PointSet,
    amplitude_log_density,
    grid_log_posterior,
    gwb_likelihood_model,
)
from astrogwb.inference.compound_poisson import (
    cumulants,
    inclination_quadrature,
    saddle,
)
from astrogwb.populations._types import SourceModel

# --------------------------------------------------------------------------- #
# The density, against exact and Monte Carlo references
# --------------------------------------------------------------------------- #
LATTICE_TOYS = {
    # A handful of sources: strongly skewed.
    "skewed": ([6.0, 3.0, 1.0], [1, 2, 4], 0.05, 400.0),
    # Hundreds of sources: near the Gaussian limit.
    "near_gaussian": ([400.0, 300.0, 200.0], [2, 3, 5], 0.0025, 1e4),
}


def _lattice_toy(name: str) -> tuple[PointSet, np.ndarray]:
    """A toy point set and its exact log density on a grid of standardized ``b``.

    The amplitudes are integer multiples of one step, so the count
    :math:`M = \\sum_i k_i N_i` has a pmf found by convolving each source's
    Poisson lattice, and the density is that pmf smeared by the Gaussian noise.
    """
    counts, multiples, step, precision = LATTICE_TOYS[name]
    points = PointSet(
        jnp.asarray(counts),
        jnp.asarray(multiples, dtype=float) * step,
        jnp.asarray(precision),
    )
    size = int(4 * np.dot(counts, multiples)) + 200
    pmf = np.zeros(size)
    pmf[0] = 1.0
    for count, multiple in zip(counts, multiples, strict=True):
        draws = np.arange(size // multiple)
        source = np.zeros(size)
        source[draws * multiple] = stats.poisson.pmf(draws, count)
        pmf = np.convolve(pmf, source)[:size]
    mean = float(np.dot(counts, multiples) * step)
    sd = float(np.sqrt(np.dot(counts, np.square(multiples)) * step**2 + 1 / precision))
    b = mean + sd * np.linspace(-3.0, 4.0, 9)
    noise = stats.norm.pdf(b[:, None], np.arange(size) * step, precision**-0.5)
    return points, np.stack([b, np.log(noise @ pmf)])


def _evaluate(
    points: PointSet, b: ArrayLike, method: AmplitudeDensityMethod, **kwargs: Any
) -> np.ndarray:
    density = jax.vmap(lambda x: amplitude_log_density(x, points, method, **kwargs))
    return np.asarray(density(jnp.asarray(b)))


@pytest.mark.parametrize(
    ("toy", "num_nodes", "atol"),
    [("near_gaussian", 8, 1e-8), ("skewed", 32, 1e-5)],
)
def test_hermite_log_density_matches_the_exact_lattice_mixture(
    toy: str, num_nodes: int, atol: float
) -> None:
    points, (b, expected) = _lattice_toy(toy)

    result = _evaluate(points, b, "hermite", num_hermite_nodes=num_nodes)

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=atol)


def test_hermite_log_density_converges_with_the_node_count() -> None:
    points, (b, expected) = _lattice_toy("skewed")

    errors = [
        np.max(np.abs(_evaluate(points, b, "hermite", num_hermite_nodes=n) - expected))
        for n in (8, 16, 32)
    ]

    assert errors[0] > errors[1] > errors[2]


def test_saddlepoint_log_density_matches_hermite_near_the_gaussian_limit() -> None:
    points, (b, _) = _lattice_toy("near_gaussian")

    saddlepoint = _evaluate(points, b, "saddlepoint")
    hermite = _evaluate(points, b, "hermite", num_hermite_nodes=8)

    np.testing.assert_allclose(saddlepoint, hermite, rtol=0.0, atol=1e-4)


def test_saddlepoint_log_density_beats_the_gaussian_when_skewed() -> None:
    points, (b, expected) = _lattice_toy("skewed")

    errors = {
        method: np.max(np.abs(_evaluate(points, b, method) - expected))
        for method in ("saddlepoint", "gaussian")
    }

    assert errors["saddlepoint"] < 0.05
    assert errors["gaussian"] > 1.0


@pytest.mark.parametrize("method", ["saddlepoint", "hermite"])
def test_amplitude_log_density_integrates_to_one(
    method: AmplitudeDensityMethod,
) -> None:
    points, _ = _lattice_toy("near_gaussian")
    mean, variance = cumulants(0.0, points)[1:]
    grid = mean + jnp.sqrt(variance) * jnp.linspace(-12.0, 16.0, 4001)

    density = np.exp(_evaluate(points, grid, method))

    assert float(np.trapezoid(density, np.asarray(grid))) == pytest.approx(
        1.0, abs=1e-6
    )


def test_hermite_cdf_matches_a_monte_carlo_draw() -> None:
    """Off the lattice: P(B < x) at a few quantiles, within five binomial sigma."""
    rng = np.random.default_rng(1)
    counts = rng.uniform(0.5, 4.0, 12)
    amplitudes = rng.uniform(0.01, 0.2, 12)
    precision = 2500.0
    points = PointSet(
        jnp.asarray(counts), jnp.asarray(amplitudes), jnp.asarray(precision)
    )
    num_draws = 400_000
    draws = rng.poisson(counts, (num_draws, counts.size)) @ amplitudes
    draws = draws + rng.normal(0.0, precision**-0.5, num_draws)
    thresholds = np.quantile(draws, [0.05, 0.3, 0.6, 0.9])
    grid = np.linspace(draws.min() - 0.5, thresholds.max(), 6001)

    density = np.exp(_evaluate(points, grid, "hermite", num_hermite_nodes=32))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]))])
    cdf = cdf * (grid[1] - grid[0])
    predicted = np.interp(thresholds, grid, cdf)

    empirical = (draws[:, None] < thresholds).mean(axis=0)
    tolerance = 5.0 * np.sqrt(empirical * (1.0 - empirical) / num_draws)
    assert np.all(np.abs(predicted - empirical) < tolerance)


@pytest.mark.parametrize("offset", [-3.0, 0.0, 6.0])
def test_saddle_solves_the_derivative_equation(offset: float) -> None:
    points, _ = _lattice_toy("skewed")
    mean, variance = cumulants(0.0, points)[1:]
    b = mean + offset * jnp.sqrt(variance)

    t = saddle(b, points)

    assert float(cumulants(t, points)[1]) == pytest.approx(float(b), rel=1e-12)


@pytest.mark.parametrize("method", ["saddlepoint", "hermite"])
@pytest.mark.parametrize("order", [1, 2])
def test_amplitude_log_density_derivatives_match_finite_differences(
    method: AmplitudeDensityMethod, order: int
) -> None:
    """The saddle is a custom root: first and second derivatives are exact."""
    points, (b, _) = _lattice_toy("near_gaussian")

    def log_density(scale: ArrayLike) -> jax.Array:
        scaled = points._replace(amplitudes=points.amplitudes * scale)
        return amplitude_log_density(b[6], scaled, method)

    function = jax.grad(log_density) if order == 2 else log_density
    analytic = jax.grad(function)(1.0)
    step = 1e-5
    numeric = (function(1.0 + step) - function(1.0 - step)) / (2 * step)
    assert float(analytic) == pytest.approx(float(numeric), rel=1e-6)


@pytest.mark.parametrize("num_nodes", [5, 8, 24])
def test_inclination_quadrature_has_the_exact_first_two_moments(
    num_nodes: int,
) -> None:
    pattern, weights = inclination_quadrature(num_nodes)

    assert weights.sum() == pytest.approx(1.0)
    assert weights @ pattern == pytest.approx(1.0)
    assert weights @ pattern**2 == pytest.approx(355.0 / 252.0)


# --------------------------------------------------------------------------- #
# ImportanceCompoundPoissonLikelihood on the mock catalog
# --------------------------------------------------------------------------- #
_TARGET = mock_population()
NUM_NETWORKS = 3
OBSERVATION_TIME = 2.0


def pinned_target(
    params: Mapping[str, ArrayLike],
) -> tuple[RedshiftDistribution, SourceModel]:
    """The target population, unsampled hyperparameters pinned at the fiducials."""
    return _TARGET({**FIDUCIALS, **params})


@pytest.fixture(scope="module")
def catalog() -> Any:
    return build_reference_catalog(num_sources=16)


@pytest.fixture(scope="module")
def frequencies(catalog: Any) -> np.ndarray:
    return catalog[0]["frequencies"][::4][:12]


@pytest.fixture(scope="module")
def observed(catalog: Any, frequencies: np.ndarray) -> jax.Array:
    data, metadata = catalog
    reference = build_reference_spectrum(
        data, metadata, frequencies=frequencies, population=pinned_target
    )
    return 1.05 * reference.spectrum(FIDUCIALS)


@pytest.fixture(scope="module")
def networks(observed: jax.Array) -> Network:
    """Three stacked networks: two scales, and one with masked bins."""
    rng = np.random.default_rng(0)
    mask = np.ones((NUM_NETWORKS, 12), dtype=bool)
    mask[2, ::3] = False
    typical = float(jnp.max(observed))
    return Network(
        scale=jnp.asarray(rng.uniform(0.05, 0.2, (NUM_NETWORKS, 12))) * typical,
        mask=jnp.asarray(mask),
    )


def _single(networks: Network, index: int) -> Network:
    return jax.tree_util.tree_map(lambda leaf: leaf[index], networks)


@pytest.fixture(scope="module")
def build(catalog: Any, frequencies: np.ndarray, observed: jax.Array) -> Any:
    """``(observation_time, **options) -> ImportanceCompoundPoissonLikelihood``."""
    data, metadata = catalog

    def factory(
        observation_time: float = OBSERVATION_TIME, **options: Any
    ) -> ImportanceCompoundPoissonLikelihood:
        return ImportanceCompoundPoissonLikelihood.from_catalog(
            data,
            metadata,
            population=pinned_target,
            frequencies=frequencies,
            num_redshift_nodes=NUM_REDSHIFT_NODES,
            density_sites=INTRINSIC_DENSITY_SITES,
            observation_time=observation_time,
            observed=observed,
            **options,
        )

    return factory


@pytest.fixture(scope="module")
def build_gaussian(catalog: Any, frequencies: np.ndarray, observed: jax.Array) -> Any:
    """``observation_time -> ImportanceGaussianLikelihood`` in the amplitude mode."""
    data, metadata = catalog

    def factory(observation_time: float) -> ImportanceGaussianLikelihood:
        return build_reference_spectrum(
            data,
            metadata,
            frequencies=frequencies,
            population=pinned_target,
            observed=observed,
            shot_noise="amplitude",
            observation_time=observation_time,
        )

    return factory


@pytest.mark.parametrize("index", range(NUM_NETWORKS))
def test_point_set_has_unit_mean(build: Any, networks: Network, index: int) -> None:
    weights, amplitudes, _ = build().point_set(FIDUCIALS, _single(networks, index))

    assert float(jnp.sum(weights * amplitudes)) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("index", range(NUM_NETWORKS))
def test_point_set_variance_is_the_gaussian_relative_variance(
    build: Any, build_gaussian: Any, networks: Network, index: int
) -> None:
    network = _single(networks, index)
    weights, amplitudes, _ = build().point_set(FIDUCIALS, network)

    expected = build_gaussian(OBSERVATION_TIME).relative_variance(FIDUCIALS, network)

    assert float(jnp.sum(weights * amplitudes**2)) == pytest.approx(
        float(expected), rel=1e-12
    )


@pytest.mark.parametrize("observation_time", [1e-6, OBSERVATION_TIME])
def test_gaussian_method_is_the_gaussian_amplitude_mode(
    build: Any, build_gaussian: Any, networks: Network, observation_time: float
) -> None:
    """Both apply every network in the stack to one shared prediction."""
    exact = build(observation_time, method="gaussian")
    gaussian = build_gaussian(observation_time)

    def stacked(likelihood: Any) -> jax.Array:
        return jax.vmap(
            likelihood.log_likelihood, in_axes=(None, 0), out_axes=(0, None)
        )(FIDUCIALS, networks)[0]

    np.testing.assert_allclose(stacked(exact), stacked(gaussian), rtol=1e-12)


@pytest.mark.parametrize("method", ["saddlepoint", "hermite"])
def test_log_likelihood_reduces_to_the_gaussian_for_a_large_count(
    build: Any, build_gaussian: Any, networks: Network, method: str
) -> None:
    """A large observation time makes every amplitude tiny: the Gaussian limit."""
    network = _single(networks, 0)
    exact = build(1e4, method=method).log_likelihood(FIDUCIALS, network)[0]
    gaussian = build_gaussian(1e4).log_likelihood(FIDUCIALS, network)[0]

    assert float(exact) == pytest.approx(float(gaussian), rel=1e-12)


def test_log_likelihood_departs_from_the_gaussian_for_a_small_count(
    build: Any, build_gaussian: Any, networks: Network
) -> None:
    """At :math:`10^{-4}` yr the mock catalog has ~70 sources: skew matters."""
    network = _single(networks, 0)
    exact = build(1e-4, method="hermite").log_likelihood(FIDUCIALS, network)[0]
    gaussian = build_gaussian(1e-4).log_likelihood(FIDUCIALS, network)[0]

    assert abs(float(exact - gaussian)) > 1e-3


def test_saddlepoint_matches_hermite_on_the_mock_catalog(
    build: Any, networks: Network
) -> None:
    network = _single(networks, 1)
    values = {
        method: build(1e-3, method=method, num_hermite_nodes=16).log_likelihood(
            FIDUCIALS, network
        )[0]
        for method in ("saddlepoint", "hermite")
    }

    assert float(values["saddlepoint"]) == pytest.approx(
        float(values["hermite"]), abs=1e-3
    )


@pytest.mark.parametrize("method", ["saddlepoint", "hermite"])
def test_vmap_over_networks_equals_one_evaluation_per_network(
    build: Any, networks: Network, method: str
) -> None:
    likelihood = build(1e-3, method=method)
    params = {**FIDUCIALS, "H0": 70.0}

    stacked, extras = jax.vmap(
        likelihood.log_likelihood, in_axes=(None, 0), out_axes=(0, None)
    )(params, networks)

    singles = [
        likelihood.log_likelihood(params, _single(networks, k))[0]
        for k in range(NUM_NETWORKS)
    ]
    np.testing.assert_allclose(stacked, jnp.stack(singles), rtol=1e-10)
    assert set(extras) == {"total_merger_rate", "importance_relative_ess"}


@pytest.mark.parametrize("method", ["saddlepoint", "hermite"])
def test_log_likelihood_gradient_matches_finite_differences(
    build: Any, networks: Network, method: str
) -> None:
    likelihood = build(1e-3, method=method)
    network = _single(networks, 0)

    def log_likelihood(hubble: ArrayLike) -> jax.Array:
        return likelihood.log_likelihood({**FIDUCIALS, "H0": hubble}, network)[0]

    # Off the peak at the fiducial, where the gradient is not round-off.
    analytic = jax.jit(jax.grad(log_likelihood))(64.0)
    step = 1e-4
    numeric = (log_likelihood(64.0 + step) - log_likelihood(64.0 - step)) / (2 * step)
    assert bool(jnp.isfinite(analytic))
    assert float(analytic) == pytest.approx(float(numeric), rel=1e-5)


def test_grid_log_posterior_matches_the_numpyro_log_density_per_network(
    build: Any, networks: Network
) -> None:
    priors = {"H0": dist.Uniform(20.0, 140.0)}
    grids = {"H0": jnp.linspace(66.0, 70.0, 3)}
    likelihood = build(1e-3)

    result = grid_log_posterior(likelihood, priors, grids, networks=networks)

    density = LogDensityFn(partial(gwb_likelihood_model, priors=priors))
    expected = jnp.stack(
        [
            density(
                grids,
                likelihood=dataclasses.replace(
                    likelihood, network=_single(networks, k)
                ),
            )
            for k in range(NUM_NETWORKS)
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10)


@pytest.mark.parametrize("index", range(NUM_NETWORKS))
def test_point_set_under_a_network_response_has_the_gaussian_moments(
    build: Any, build_gaussian: Any, networks: Network, index: int
) -> None:
    """The response nodes replace the inclination nodes in both likelihoods alike."""
    network = _single(networks, index)._replace(
        response=jnp.array([0.2, 0.7, 1.1, 2.0])
    )
    weights, amplitudes, _ = build().point_set(FIDUCIALS, network)

    expected = build_gaussian(OBSERVATION_TIME).relative_variance(FIDUCIALS, network)

    assert weights.shape[-1] == 4
    assert float(jnp.sum(weights * amplitudes)) == pytest.approx(1.0, rel=1e-12)
    assert float(jnp.sum(weights * amplitudes**2)) == pytest.approx(
        float(expected), rel=1e-12
    )
