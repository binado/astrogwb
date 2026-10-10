"""The pure likelihood classes, their vmap over networks, and their grid map.

The catalog-bound tests use the closed-form mock reference catalog of
``astrogwb_mock_population``; the helper tests are closed-form on small arrays.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    build_reference_catalog,
    build_reference_spectrum,
    mock_population,
)
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer.util import log_density

from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.inference import (
    GaussianLikelihood,
    ImportanceGaussianLikelihood,
    LogDensityFn,
    Network,
    ShotNoiseMode,
    gaussian_log_likelihood,
    grid_log_posterior,
    gwb_likelihood_model,
    shot_noise_log_likelihood,
)
from astrogwb.populations._types import SourceModel

_TARGET = mock_population()

NUM_NETWORKS = 3
OBSERVATION_TIME = 2.0

MODES = [None, "amplitude", "per_frequency", "fixed"]


def pinned_target(
    params: Mapping[str, ArrayLike],
) -> tuple[RedshiftDistribution, SourceModel]:
    """The target population, unsampled hyperparameters pinned at the fiducials."""
    return _TARGET({**FIDUCIALS, **params})


# --------------------------------------------------------------------------- #
# The helpers, against closed-form values
# --------------------------------------------------------------------------- #
@pytest.fixture
def mean() -> jax.Array:
    return jnp.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def observed() -> jax.Array:
    return jnp.array([1.2, 1.7, 3.5, 3.9])


@pytest.fixture
def scale() -> jax.Array:
    return jnp.array([0.5, 1.0, 0.8, 2.0])


def _dense_log_density(
    mean: jax.Array,
    observed: jax.Array,
    covariance: jax.Array,
    keep: jax.Array | None = None,
) -> float:
    selection = np.ones(mean.shape, dtype=bool) if keep is None else np.asarray(keep)
    return float(
        dist.MultivariateNormal(
            mean[selection], np.asarray(covariance)[selection][:, selection]
        )
        .log_prob(observed[selection])
        .sum()
    )


def test_gaussian_log_likelihood_matches_the_closed_form(
    mean: jax.Array, observed: jax.Array, scale: jax.Array
) -> None:
    expected = -0.5 * np.sum(
        ((observed - mean) / scale) ** 2 + np.log(2.0 * np.pi * scale**2)
    )
    np.testing.assert_allclose(
        gaussian_log_likelihood(mean, observed, scale), expected, rtol=1e-12
    )


def test_gaussian_log_likelihood_drops_masked_bins_even_with_infinite_scale(
    mean: jax.Array, observed: jax.Array, scale: jax.Array
) -> None:
    mask = jnp.array([True, False, True, True])
    infinite = jnp.where(mask, scale, jnp.inf)

    def total(m: jax.Array) -> jax.Array:
        return gaussian_log_likelihood(m, observed, infinite, mask)

    kept = np.asarray(mask)
    np.testing.assert_allclose(
        total(mean),
        gaussian_log_likelihood(mean[kept], observed[kept], scale[kept]),
        rtol=1e-12,
    )
    assert bool(jnp.all(jnp.isfinite(jax.grad(total)(mean))))


@pytest.mark.parametrize("mode", ["amplitude", "per_frequency", "fixed"])
def test_shot_noise_with_zero_variance_is_the_diagonal_gaussian(
    mean: jax.Array,
    observed: jax.Array,
    scale: jax.Array,
    mode: ShotNoiseMode,
) -> None:
    network = Network(scale, relative_variance=jnp.zeros(()))
    result = shot_noise_log_likelihood(
        mean, jnp.zeros_like(mean), observed, network, mode
    )
    np.testing.assert_allclose(
        result, gaussian_log_likelihood(mean, observed, scale), rtol=1e-10
    )


def test_fixed_shot_noise_matches_the_dense_covariance(
    mean: jax.Array, observed: jax.Array, scale: jax.Array
) -> None:
    relative_variance = jnp.asarray(0.04)
    mask = jnp.array([True, True, False, True])
    network = Network(scale, mask, relative_variance)

    covariance = jnp.diag(scale**2) + relative_variance * jnp.outer(mean, mean)
    np.testing.assert_allclose(
        shot_noise_log_likelihood(mean, None, observed, network, "fixed"),
        _dense_log_density(mean, observed, covariance, mask),
        rtol=1e-10,
    )


def test_per_frequency_shot_noise_matches_the_dense_covariance(
    mean: jax.Array, observed: jax.Array, scale: jax.Array
) -> None:
    variance = (0.1 * mean * jnp.array([1.0, 1.4, 0.7, 1.1])) ** 2
    direction = jnp.sqrt(variance)

    covariance = jnp.diag(scale**2) + jnp.outer(direction, direction)
    np.testing.assert_allclose(
        shot_noise_log_likelihood(
            mean, variance, observed, Network(scale), "per_frequency"
        ),
        _dense_log_density(mean, observed, covariance),
        rtol=1e-10,
    )


def test_amplitude_shot_noise_with_flat_relative_scatter_is_the_fixed_mode(
    mean: jax.Array, observed: jax.Array, scale: jax.Array
) -> None:
    """A flat :math:`\\sqrt{V}/S` makes the amplitude mode exact: :math:`s = 0.1`."""
    variance = (0.1 * mean) ** 2
    amplitude = shot_noise_log_likelihood(
        mean, variance, observed, Network(scale), "amplitude"
    )
    fixed = shot_noise_log_likelihood(
        mean, None, observed, Network(scale, None, jnp.asarray(0.01)), "fixed"
    )
    np.testing.assert_allclose(amplitude, fixed, rtol=1e-10)


# --------------------------------------------------------------------------- #
# GaussianLikelihood
# --------------------------------------------------------------------------- #
def _analytic(params: Mapping[str, ArrayLike]) -> tuple[jax.Array, dict[str, Any]]:
    mean = jnp.asarray(params["h0"]) * jnp.array([1.0, 1.5, 2.0])
    return mean, {"total": jnp.sum(mean)}


@pytest.fixture
def analytic(observed: jax.Array) -> GaussianLikelihood:
    return GaussianLikelihood(
        _analytic, observed[:3], Network(jnp.array([0.7, 0.9, 1.2]))
    )


def test_gaussian_likelihood_returns_the_value_and_the_extras(
    analytic: GaussianLikelihood,
) -> None:
    value, extras = analytic({"h0": 1.1})

    assert value.shape == ()
    assert set(extras) == {"total"}
    np.testing.assert_allclose(extras["total"], 1.1 * 4.5, rtol=1e-12)


def test_gaussian_likelihood_accepts_a_spectrum_without_extras(
    analytic: GaussianLikelihood,
) -> None:
    bare = dataclasses.replace(
        analytic, spectrum_fn=lambda params: _analytic(params)[0]
    )
    value, extras = bare({"h0": 1.1})

    np.testing.assert_allclose(value, analytic({"h0": 1.1})[0], rtol=1e-12)
    assert extras == {}


def test_gwb_likelihood_model_records_priors_extras_and_one_factor(
    analytic: GaussianLikelihood,
) -> None:
    model = partial(
        gwb_likelihood_model,
        likelihood=analytic,
        priors={"h0": dist.Uniform(0.5, 2.0)},
    )
    trace = handlers.trace(handlers.seed(model, 0)).get_trace()

    assert set(trace) == {"h0", "total", "log_likelihood"}
    assert trace["h0"]["type"] == "sample"
    assert trace["total"]["type"] == "deterministic"
    assert trace["log_likelihood"]["fn"].__class__.__name__ == "Unit"


def test_gwb_likelihood_model_density_is_log_prior_plus_likelihood(
    analytic: GaussianLikelihood,
) -> None:
    prior = dist.Uniform(0.5, 2.0)
    model = partial(gwb_likelihood_model, likelihood=analytic, priors={"h0": prior})

    log_joint, _ = log_density(model, (), {}, {"h0": jnp.asarray(1.1)})

    np.testing.assert_allclose(
        log_joint,
        prior.log_prob(1.1) + analytic({"h0": 1.1})[0],
        rtol=1e-12,
    )


# --------------------------------------------------------------------------- #
# ImportanceGaussianLikelihood
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def catalog() -> Any:
    return build_reference_catalog(num_sources=16)


@pytest.fixture(scope="module")
def networks(catalog: Any) -> Network:
    """Three stacked networks: plain, another scale, and one with masked bins."""
    typical = float(jnp.max(_likelihood(catalog, None).spectrum(FIDUCIALS)))
    rng = np.random.default_rng(0)
    mask = np.ones((NUM_NETWORKS, 12), dtype=bool)
    mask[2, ::3] = False
    return Network(
        scale=jnp.asarray(rng.uniform(0.05, 0.2, (NUM_NETWORKS, 12))) * typical,
        mask=jnp.asarray(mask),
        relative_variance=jnp.array([0.0, 0.01, 0.04]),
    )


def _single(networks: Network, index: int) -> Network:
    return jax.tree_util.tree_map(lambda leaf: leaf[index], networks)


def _likelihood(catalog: Any, mode: str | None, **kwargs: Any) -> Any:
    data, metadata = catalog
    frequencies = data["frequencies"][::4][:12]
    reference = build_reference_spectrum(
        data, metadata, frequencies=frequencies, population=pinned_target
    )
    observed = 1.05 * reference.spectrum(FIDUCIALS)
    return build_reference_spectrum(
        data,
        metadata,
        frequencies=frequencies,
        population=pinned_target,
        observed=observed,
        shot_noise=mode,
        observation_time=OBSERVATION_TIME,
        **kwargs,
    )


def test_from_catalog_stores_the_squared_power_only_when_a_variance_is_needed(
    catalog: Any,
) -> None:
    held = {
        mode: _likelihood(catalog, mode).squared_power is not None for mode in MODES
    }

    assert held == {
        None: False,
        "amplitude": True,
        "per_frequency": True,
        "fixed": False,
    }


def test_predict_returns_a_variance_only_for_the_modes_that_use_one(
    catalog: Any,
) -> None:
    for mode in MODES:
        mean, variance, extras = _likelihood(catalog, mode).predict(FIDUCIALS)
        assert mean.shape == (12,)
        assert (variance is not None) == (mode in ("amplitude", "per_frequency"))
        assert set(extras) == {"total_merger_rate", "importance_relative_ess"}


def test_the_importance_weights_are_one_at_the_fiducials(catalog: Any) -> None:
    likelihood = _likelihood(catalog, None)

    np.testing.assert_array_equal(
        np.asarray(likelihood.log_weights(FIDUCIALS)), np.zeros(16)
    )


@pytest.mark.parametrize("mode", MODES, ids=str)
def test_vmap_over_networks_equals_one_evaluation_per_network(
    catalog: Any, networks: Network, mode: str | None
) -> None:
    likelihood = _likelihood(catalog, mode)
    params = {**FIDUCIALS, "H0": 70.0}

    stacked, extras = jax.vmap(
        likelihood.log_likelihood, in_axes=(None, 0), out_axes=(0, None)
    )(params, networks)

    singles = [
        likelihood.log_likelihood(params, _single(networks, k))[0]
        for k in range(NUM_NETWORKS)
    ]
    np.testing.assert_allclose(stacked, jnp.stack(singles), rtol=1e-10)
    # Unbatched extras: the prediction does not depend on the network.
    assert all(value.shape == () for value in extras.values())


def _walk(jaxpr: Any) -> Iterator[Any]:
    """Every equation of ``jaxpr``, recursing into its sub-jaxprs."""
    for equation in jaxpr.eqns:
        yield equation
        for value in equation.params.values():
            for candidate in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(candidate, "jaxpr", candidate)
                if hasattr(inner, "eqns"):
                    yield from _walk(inner)


@pytest.mark.parametrize("mode", ["amplitude", None], ids=str)
def test_the_catalog_contraction_is_shared_across_networks(
    catalog: Any, networks: Network, mode: str | None
) -> None:
    likelihood = _likelihood(catalog, mode)
    power_shape = likelihood.power.shape

    jaxpr = jax.make_jaxpr(
        jax.vmap(likelihood.log_likelihood, in_axes=(None, 0), out_axes=(0, None))
    )(FIDUCIALS, networks)

    contractions = [
        [tuple(operand.aval.shape) for operand in equation.invars]
        for equation in _walk(jaxpr.jaxpr)
        if equation.primitive.name == "dot_general"
        and any(tuple(operand.aval.shape) == power_shape for operand in equation.invars)
    ]
    num_power_operands = 2 if mode == "amplitude" else 1
    assert len(contractions) == num_power_operands
    for shapes in contractions:
        assert sorted(shapes) == sorted([power_shape, (power_shape[1],)])


def test_a_catalog_is_a_traced_input_that_compiles_once_per_shape(
    catalog: Any,
) -> None:
    traces: list[None] = []

    @jax.jit
    def spectrum(likelihood: ImportanceGaussianLikelihood, params: Any) -> jax.Array:
        traces.append(None)
        return likelihood.spectrum(params)

    first = _likelihood(catalog, None)
    second = dataclasses.replace(first, power=1.1 * first.power)
    results = [np.asarray(spectrum(lik, FIDUCIALS)) for lik in (first, second)]

    assert len(traces) == 1
    assert not np.allclose(results[0], results[1], rtol=1e-6, atol=0.0)

    smaller = _likelihood(build_reference_catalog(num_sources=8), None)
    spectrum(smaller, FIDUCIALS)
    assert len(traces) == 2


# --------------------------------------------------------------------------- #
# The grid map
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("grids", "fixed"),
    [
        ({"H0": jnp.linspace(66.0, 70.0, 5)}, {"Omega_m": jnp.asarray(0.31)}),
        (
            {"H0": jnp.linspace(66.0, 70.0, 3), "Omega_m": jnp.linspace(0.3, 0.32, 2)},
            {},
        ),
    ],
    ids=["1d", "2d"],
)
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_grid_log_posterior_matches_the_numpyro_log_density_per_network(
    catalog: Any,
    networks: Network,
    grids: dict[str, jax.Array],
    fixed: dict[str, jax.Array],
    chunk_size: int | None,
) -> None:
    priors = {
        "H0": dist.Uniform(20.0, 140.0),
        "Omega_m": dist.Normal(0.3096, 0.006),
    }
    likelihood = _likelihood(catalog, None)

    result = grid_log_posterior(
        likelihood,
        priors,
        grids,
        fixed=fixed,
        networks=networks,
        chunk_size=chunk_size,
    )

    density = LogDensityFn(partial(gwb_likelihood_model, priors=priors))
    expected = jnp.stack(
        [
            density(
                grids,
                fixed=fixed,
                likelihood=dataclasses.replace(
                    likelihood, network=_single(networks, k)
                ),
            )
            for k in range(NUM_NETWORKS)
        ]
    )
    assert result.shape == (NUM_NETWORKS, *(grid.size for grid in grids.values()))
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_grid_log_posterior_outside_the_prior_support_is_negative_infinite(
    catalog: Any, networks: Network
) -> None:
    result = grid_log_posterior(
        _likelihood(catalog, None),
        {"H0": dist.Uniform(20.0, 60.0)},
        {"H0": jnp.array([40.0, 70.0])},
        networks=networks,
    )

    assert bool(jnp.all(jnp.isfinite(result[:, 0])))
    assert bool(jnp.all(result[:, 1] == -jnp.inf))
