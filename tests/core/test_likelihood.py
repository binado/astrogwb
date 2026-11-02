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
from numpyro.infer.util import log_density

from astrogwb.background.importance import INCLINATION_SECOND_MOMENT
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.inference import (
    GaussianLikelihood,
    ImportanceGaussianLikelihood,
    LogDensityFn,
    Network,
    amplitude_shot_noise_variance,
    gaussian_log_likelihood,
    grid_log_posterior,
    gwb_likelihood_model,
    shot_noise_log_likelihood,
    shot_noise_modes,
)
from astrogwb.populations._types import SourceModel

_TARGET = mock_population()

NUM_NETWORKS = 3
#: The network with masked bins: it exercises every branch the others do.
MASKED = 2
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


def test_gaussian_likelihood_accepts_a_spectrum_without_extras(
    analytic: GaussianLikelihood,
) -> None:
    bare = dataclasses.replace(
        analytic, spectrum_fn=lambda params: _analytic(params)[0]
    )
    value, extras = bare({"h0": 1.1})

    np.testing.assert_allclose(value, analytic({"h0": 1.1})[0], rtol=1e-12)
    assert extras == {}


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


@pytest.fixture(scope="module")
def amplitude_likelihood(catalog: Any) -> ImportanceGaussianLikelihood:
    return _likelihood(catalog, "amplitude")


def test_relative_variance_is_bounded_by_the_snr_weighted_scatter(
    amplitude_likelihood: ImportanceGaussianLikelihood, networks: Network
) -> None:
    """Cauchy-Schwarz: the projected variance never exceeds the per-bin bound."""
    network = _single(networks, MASKED)
    mean, variance, _ = amplitude_likelihood.predict(FIDUCIALS)
    assert variance is not None
    bound = amplitude_shot_noise_variance(mean, variance, network.scale, network.mask)

    assert float(amplitude_likelihood.relative_variance(FIDUCIALS, network)) <= float(
        bound
    )


@pytest.mark.parametrize("bin_index", [3, 11])
def test_relative_variance_of_one_counted_bin_is_its_fractional_variance(
    amplitude_likelihood: ImportanceGaussianLikelihood, bin_index: int
) -> None:
    """One bin projects on itself; only the interpolation order of the square differs."""
    mean, variance, _ = amplitude_likelihood.predict(FIDUCIALS)
    assert variance is not None
    mask = jnp.zeros(mean.shape, dtype=bool).at[bin_index].set(True)
    network = Network(jnp.ones_like(mean), mask)

    assert float(
        amplitude_likelihood.relative_variance(FIDUCIALS, network)
    ) == pytest.approx(float(variance[bin_index] / mean[bin_index] ** 2), rel=1e-2)


def test_catalog_covariance_diagonal_is_bounded_by_the_per_bin_variance(
    amplitude_likelihood: ImportanceGaussianLikelihood,
) -> None:
    """Squaring after interpolation never exceeds interpolating the square."""
    _, variance, _ = amplitude_likelihood.predict(FIDUCIALS)
    assert variance is not None
    diagonal = jnp.diag(amplitude_likelihood.catalog_covariance(FIDUCIALS))

    assert bool(jnp.all(diagonal <= variance * (1.0 + 1e-12)))


def test_catalog_covariance_is_symmetric_positive_semidefinite(
    amplitude_likelihood: ImportanceGaussianLikelihood,
) -> None:
    covariance = np.asarray(amplitude_likelihood.catalog_covariance(FIDUCIALS))
    eigenvalues = np.linalg.eigvalsh(covariance)

    np.testing.assert_array_equal(covariance, covariance.T)
    assert eigenvalues.min() >= -1e-10 * eigenvalues.max()


def test_amplitude_log_likelihood_is_the_fixed_mode_at_the_relative_variance(
    catalog: Any, networks: Network
) -> None:
    network = _single(networks, 2)
    params = {**FIDUCIALS, "H0": 70.0}
    amplitude = _likelihood(catalog, "amplitude")
    pinned = network._replace(
        relative_variance=amplitude.relative_variance(params, network)
    )

    np.testing.assert_allclose(
        amplitude.log_likelihood(params, network)[0],
        _likelihood(catalog, "fixed").log_likelihood(params, pinned)[0],
        rtol=1e-12,
    )


def test_the_importance_weights_are_one_at_the_fiducials(catalog: Any) -> None:
    likelihood = _likelihood(catalog, None)

    np.testing.assert_array_equal(
        np.asarray(likelihood.log_weights(FIDUCIALS)), np.zeros(16)
    )


@pytest.mark.parametrize("mode", MODES, ids=str)
def test_each_network_scores_its_own_observed_spectrum(
    catalog: Any, networks: Network, mode: str | None
) -> None:
    likelihood = _likelihood(catalog, mode)
    params = {**FIDUCIALS, "H0": 70.0}
    observed = likelihood.observed * jnp.array([[1.0], [0.97], [1.02]])

    stacked, _ = jax.vmap(
        likelihood.log_likelihood, in_axes=(None, 0), out_axes=(0, None)
    )(params, networks._replace(observed=observed))

    singles = [
        dataclasses.replace(likelihood, observed=observed[k]).log_likelihood(
            params, _single(networks, k)
        )[0]
        for k in range(NUM_NETWORKS)
    ]
    np.testing.assert_allclose(stacked, jnp.stack(singles), rtol=1e-10)


def _walk(jaxpr: Any) -> Iterator[Any]:
    """Every equation of ``jaxpr``, recursing into its sub-jaxprs."""
    for equation in jaxpr.eqns:
        yield equation
        for value in equation.params.values():
            for candidate in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(candidate, "jaxpr", candidate)
                if hasattr(inner, "eqns"):
                    yield from _walk(inner)


@pytest.mark.parametrize("mode", ["amplitude", "per_frequency"])
def test_the_held_mean_power_matches_the_weighted_mean(
    catalog: Any, networks: Network, mode: str | None
) -> None:
    held = _likelihood(catalog, mode, density_sites=())
    weighted = dataclasses.replace(
        held, mean_power=None, mean_squared_power=None, gram=None
    )
    params = {**FIDUCIALS, "H0": 70.0}
    network = _single(networks, 1)

    for got, want in zip(held.predict(params)[:2], weighted.predict(params)[:2]):
        if want is None:
            assert got is None
        else:
            np.testing.assert_allclose(got, want, rtol=1e-10)
    np.testing.assert_allclose(
        held.log_likelihood(params, network)[0],
        weighted.log_likelihood(params, network)[0],
        rtol=1e-10,
    )


@pytest.mark.parametrize("mode", ["per_frequency", "amplitude"])
def test_the_catalog_is_not_contracted_per_evaluation_without_density_sites(
    catalog: Any, networks: Network, mode: str | None
) -> None:
    likelihood = _likelihood(catalog, mode, density_sites=())
    power_shape = likelihood.power.shape

    jaxpr = jax.make_jaxpr(likelihood.log_likelihood)(FIDUCIALS, _single(networks, 1))

    assert not [
        equation
        for equation in _walk(jaxpr.jaxpr)
        if equation.primitive.name == "dot_general"
        and any(tuple(operand.aval.shape) == power_shape for operand in equation.invars)
    ]


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
def test_grid_log_posterior_matches_the_numpyro_log_density_per_network(
    catalog: Any,
    networks: Network,
    grids: dict[str, jax.Array],
    fixed: dict[str, jax.Array],
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


def _flat_response(values: jax.Array, num_bins: int = 12) -> jax.Array:
    """Response draws constant across the band, ``(L, F)``."""
    return jnp.broadcast_to(values[:, None], (values.shape[0], num_bins))


@pytest.fixture(scope="module")
def varying_response() -> jax.Array:
    """Seeded response draws that vary across the band, with unit mean, ``(64, 12)``."""
    draws = jax.random.gamma(jax.random.key(0), 3.0, (64, 12)) / 3.0
    return draws / jnp.mean(draws, axis=0)


#: Flat draws with :math:`\langle r^2 \rangle` equal to the inclination factor.
INCLINATION_LIKE = 1.0 + jnp.array([-1.0, 1.0]) * np.sqrt(
    INCLINATION_SECOND_MOMENT - 1.0
)


def test_a_flat_response_scales_the_relative_variance_by_its_second_moment(
    amplitude_likelihood: ImportanceGaussianLikelihood, networks: Network
) -> None:
    """Flat in frequency, the response is a scalar inflation."""
    values = jnp.array([0.2, 0.7, 1.1, 2.0])
    network = _single(networks, MASKED)

    ratio = amplitude_likelihood.relative_variance(
        FIDUCIALS, network._replace(response=_flat_response(values))
    ) / amplitude_likelihood.relative_variance(FIDUCIALS, network)

    assert float(ratio) == pytest.approx(
        float(jnp.mean(values**2)) / INCLINATION_SECOND_MOMENT, rel=1e-12
    )


def test_the_relative_variance_is_the_hadamard_covariance_along_the_projector(
    amplitude_likelihood: ImportanceGaussianLikelihood,
    networks: Network,
    varying_response: jax.Array,
) -> None:
    """The projected path and the dense Hadamard covariance agree draw by draw."""
    network = _single(networks, MASKED)._replace(response=varying_response)
    assert network.mask is not None
    mean = amplitude_likelihood.spectrum(FIDUCIALS)
    projector = jnp.where(network.mask, mean / network.scale**2, 0.0)
    covariance = amplitude_likelihood.catalog_covariance(FIDUCIALS, network)

    assert float(
        projector @ covariance @ projector / (projector @ mean) ** 2
    ) == pytest.approx(
        float(amplitude_likelihood.relative_variance(FIDUCIALS, network)), rel=1e-10
    )


def test_held_response_moments_give_the_relative_variance_of_the_draws(
    amplitude_likelihood: ImportanceGaussianLikelihood,
    networks: Network,
    varying_response: jax.Array,
) -> None:
    network = _single(networks, MASKED)._replace(response=varying_response)
    held = network._replace(
        response_moments=amplitude_likelihood.response_moments(FIDUCIALS, network)
    )
    params = {**FIDUCIALS, "H0": 70.0}

    assert float(amplitude_likelihood.relative_variance(params, held)) == (
        pytest.approx(
            float(amplitude_likelihood.relative_variance(params, network)), rel=1e-10
        )
    )


def test_an_inclination_like_flat_response_leaves_the_per_frequency_mode_unchanged(
    catalog: Any, networks: Network
) -> None:
    likelihood = _likelihood(catalog, "per_frequency")
    network = _single(networks, 1)
    responsive = network._replace(response=_flat_response(INCLINATION_LIKE))

    assert float(likelihood.log_likelihood(FIDUCIALS, responsive)[0]) == pytest.approx(
        float(likelihood.log_likelihood(FIDUCIALS, network)[0]), rel=1e-12
    )


# --------------------------------------------------------------------------- #
# The k-mode shot noise
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def modes_likelihood(catalog: Any) -> ImportanceGaussianLikelihood:
    return _likelihood(catalog, "modes")


def _with_modes(
    likelihood: ImportanceGaussianLikelihood, network: Network, **options: Any
) -> Network:
    """``network`` with the leading eigenmodes of its catalog covariance."""
    covariance = likelihood.catalog_covariance(FIDUCIALS, network)
    return network._replace(
        modes=shot_noise_modes(covariance, network.scale, network.mask, **options)
    )


def _dense_log_likelihood(
    likelihood: ImportanceGaussianLikelihood, network: Network
) -> float:
    """The Gaussian with the detector noise plus the full catalog covariance."""
    assert network.mask is not None and likelihood.observed is not None
    keep = np.asarray(network.mask)
    mean = np.asarray(likelihood.spectrum(FIDUCIALS))[keep]
    covariance = np.asarray(likelihood.catalog_covariance(FIDUCIALS, network))[
        np.ix_(keep, keep)
    ] + np.diag(np.asarray(network.scale)[keep] ** 2)
    return float(
        dist.MultivariateNormal(jnp.asarray(mean), jnp.asarray(covariance)).log_prob(
            likelihood.observed[keep]
        )
    )


@pytest.mark.parametrize("with_response", [False, True])
def test_every_mode_gives_the_dense_gaussian_with_the_catalog_covariance(
    modes_likelihood: ImportanceGaussianLikelihood,
    networks: Network,
    varying_response: jax.Array,
    with_response: bool,
) -> None:
    network = _single(networks, MASKED)
    if with_response:
        network = network._replace(response=varying_response)
    network = _with_modes(modes_likelihood, network, tolerance=0.0, max_modes=12)

    value = modes_likelihood.log_likelihood(FIDUCIALS, network)[0]

    assert float(value) == pytest.approx(
        _dense_log_likelihood(modes_likelihood, network), rel=1e-9
    )


def test_the_template_mode_alone_is_the_amplitude_likelihood(
    catalog: Any,
    modes_likelihood: ImportanceGaussianLikelihood,
    networks: Network,
) -> None:
    """``U = e``, the whitened template direction: ``B`` is ``rho^2 s^2``."""
    network = _single(networks, MASKED)
    assert network.mask is not None
    whitened = jnp.where(
        network.mask, modes_likelihood.spectrum(FIDUCIALS) / network.scale, 0.0
    )
    template_mode = (whitened / jnp.linalg.norm(whitened))[:, None]

    value = modes_likelihood.log_likelihood(
        FIDUCIALS, network._replace(modes=template_mode)
    )[0]

    assert float(value) == pytest.approx(
        float(_likelihood(catalog, "amplitude").log_likelihood(FIDUCIALS, network)[0]),
        rel=1e-10,
    )


def test_held_mode_blocks_give_the_same_likelihood(
    modes_likelihood: ImportanceGaussianLikelihood,
    networks: Network,
    varying_response: jax.Array,
) -> None:
    network = _single(networks, MASKED)._replace(response=varying_response)
    network = _with_modes(modes_likelihood, network, max_modes=4)
    held = network._replace(
        mode_blocks=modes_likelihood.mode_blocks(FIDUCIALS, network)
    )
    shifted = {**FIDUCIALS, "H0": 1.05 * FIDUCIALS["H0"]}

    assert float(modes_likelihood.log_likelihood(shifted, held)[0]) == pytest.approx(
        float(modes_likelihood.log_likelihood(shifted, network)[0]), rel=1e-12
    )


def test_with_shot_noise_modes_pads_the_basis_and_holds_the_blocks_on_request(
    modes_likelihood: ImportanceGaussianLikelihood, networks: Network
) -> None:
    network = modes_likelihood.with_shot_noise_modes(
        FIDUCIALS, _single(networks, 0), max_modes=3, num_modes=5, hold_blocks=True
    )
    assert network.modes is not None and network.mode_blocks is not None

    assert network.modes.shape == (12, 5)
    assert network.mode_blocks.shape[1:] == (5, 5)
    np.testing.assert_array_equal(np.asarray(network.modes[:, 3:]), 0.0)


def test_with_shot_noise_modes_holds_no_blocks_while_the_weights_vary(
    modes_likelihood: ImportanceGaussianLikelihood, networks: Network
) -> None:
    """The mock catalog has intrinsic density sites, so the blocks would go stale."""
    assert modes_likelihood.density_sites

    network = modes_likelihood.with_shot_noise_modes(FIDUCIALS, _single(networks, 0))

    assert network.mode_blocks is None
