"""The catalog model, the dense Gaussian likelihood and what is derived from it.

The catalog-bound tests use the closed-form mock reference catalog of
``astrogwb_mock_population`` on a 12-bin grid and two synthetic detector pairs
whose response draws vary across the band.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from astrogwb.detector import load_detector, pair_response_moment
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.inference import (
    CatalogModel,
    DenseGaussianLikelihood,
    Prediction,
    amplitude_cumulants,
    amplitude_template,
    dense_fisher_matrix,
    fisher_matrix_per_bin,
    grid_log_likelihood,
    grid_log_prior,
    shape_correction,
    shot_noise_variance,
    whitened_catalog_covariance,
)
from astrogwb.inference import Network as StackedNetwork
from astrogwb.inference.gaussian import Network
from astrogwb.inference.model import interpolation_weights
from astrogwb.populations._types import SourceModel

_TARGET = mock_population()

NUM_PAIRS = 2
NUM_BINS = 12
NUM_DRAWS = 64
OBSERVATION_TIME = 2.0
OFF_FIDUCIAL = {**FIDUCIALS, "H0": 70.0}


def pinned_target(
    params: Mapping[str, ArrayLike],
) -> tuple[RedshiftDistribution, SourceModel]:
    """The target population, unsampled hyperparameters pinned at the fiducials."""
    return _TARGET({**FIDUCIALS, **params})


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def catalog() -> Any:
    return build_reference_catalog(num_sources=16)


@pytest.fixture(scope="module")
def frequencies(catalog: Any) -> np.ndarray:
    return np.asarray(catalog[0]["frequencies"][::4][:NUM_BINS])


@pytest.fixture(scope="module")
def model_factory(catalog: Any, frequencies: np.ndarray) -> Callable[..., CatalogModel]:
    def build(density_sites: tuple[str, ...] = ()) -> CatalogModel:
        data, metadata = catalog
        return CatalogModel.from_catalog(
            data,
            metadata,
            population=pinned_target,
            frequencies=frequencies,
            num_redshift_nodes=NUM_REDSHIFT_NODES,
            observation_time=OBSERVATION_TIME,
            density_sites=density_sites,
        )

    return build


@pytest.fixture(scope="module")
def model(model_factory: Callable[..., CatalogModel]) -> CatalogModel:
    return model_factory()


@pytest.fixture(scope="module")
def response_draws() -> jax.Array:
    """Seeded per-pair response draws that vary across the band, ``(L, P, F)``."""
    draws = jax.random.gamma(jax.random.key(0), 3.0, (NUM_DRAWS, NUM_PAIRS, NUM_BINS))
    return draws / 3.0


@pytest.fixture(scope="module")
def pair_leaves(model: CatalogModel) -> dict[str, np.ndarray]:
    """Scale, ORF and mask of two pairs; the second pair drops every third bin."""
    typical = float(jnp.max(model.spectrum(FIDUCIALS)))
    rng = np.random.default_rng(0)
    mask = np.ones((NUM_PAIRS, NUM_BINS), dtype=bool)
    mask[1, ::3] = False
    scale = rng.uniform(0.05, 0.2, (NUM_PAIRS, NUM_BINS)) * typical
    return {
        "scale": np.where(mask, scale, np.inf),
        "pair_overlap": np.stack(
            [np.linspace(0.9, 0.4, NUM_BINS), np.linspace(-0.3, 0.2, NUM_BINS)]
        ),
        "mask": mask,
    }


@pytest.fixture(scope="module")
def network(pair_leaves: dict[str, np.ndarray], response_draws: jax.Array) -> Network:
    flat = np.reshape(np.asarray(response_draws), (NUM_DRAWS, -1))
    moment = (flat.T @ flat / NUM_DRAWS).reshape((NUM_PAIRS, NUM_BINS) * 2)
    return Network.from_pairs(**pair_leaves, response_moment=moment)


@pytest.fixture(scope="module")
def observed(
    model: CatalogModel, pair_leaves: dict[str, np.ndarray], network: Network
) -> jax.Array:
    """Three realizations about the fiducial mean, on the counted entries, ``(3, D)``."""
    mean = pair_leaves["pair_overlap"] * np.asarray(model.spectrum(FIDUCIALS))
    noise = np.random.default_rng(1).normal(size=(3, NUM_PAIRS, NUM_BINS))
    scale = np.where(pair_leaves["mask"], pair_leaves["scale"], 0.0)
    return network.compress(mean * 1.05 + noise * scale)


@pytest.fixture(scope="module")
def likelihood(
    model: CatalogModel, network: Network, observed: jax.Array
) -> DenseGaussianLikelihood:
    return DenseGaussianLikelihood(model, network, observed)


# --------------------------------------------------------------------------- #
# interpolation_weights and the power moment
# --------------------------------------------------------------------------- #
def test_interpolation_weights_reproduce_jnp_interp_inside_and_outside_the_grid() -> (
    None
):
    reference = jnp.array([0.0, 1.0, 2.5, 4.0])
    query = jnp.array([-1.0, 0.0, 0.3, 1.0, 3.9, 4.0, 4.5])
    values = jax.random.normal(jax.random.key(3), (4,))
    index, weights = interpolation_weights(query, reference)

    rows = jnp.stack([index, index + 1], axis=1)
    np.testing.assert_allclose(
        jnp.sum(weights * values[rows], axis=1),
        jnp.interp(query, reference, values, right=0.0),
        rtol=1e-14,
    )


def test_power_moment_held_node_moments_match_the_weighted_gather_at_the_fiducials(
    model_factory: Callable[..., CatalogModel],
) -> None:
    """At the fiducials every weight is one, so both paths are the same moment."""
    held = model_factory()
    weighted = model_factory(INTRINSIC_DENSITY_SITES)
    assert held.node_moments is not None and weighted.node_moments is None

    np.testing.assert_allclose(
        weighted.power_moment(FIDUCIALS), held.power_moment(FIDUCIALS), rtol=1e-10
    )


def test_power_moment_is_symmetric_positive_semidefinite(model: CatalogModel) -> None:
    moment = np.asarray(model.power_moment(OFF_FIDUCIAL))
    eigenvalues = np.linalg.eigvalsh(moment)

    np.testing.assert_array_equal(moment, moment.T)
    assert eigenvalues.min() >= -1e-10 * eigenvalues.max()


def test_power_moment_matches_the_stacked_catalog_covariance(
    catalog: Any,
    frequencies: np.ndarray,
    model: CatalogModel,
    pair_leaves: dict[str, np.ndarray],
    response_draws: jax.Array,
) -> None:
    """The existing dense-interpolation covariance, under the same draws."""
    data, metadata = catalog
    stacked = build_reference_spectrum(
        data,
        metadata,
        frequencies=frequencies,
        population=pinned_target,
        density_sites=(),
        shot_noise="amplitude",
        observation_time=OBSERVATION_TIME,
    )
    old = stacked.catalog_covariance(
        OFF_FIDUCIAL,
        StackedNetwork(
            scale=jnp.asarray(pair_leaves["scale"]),
            response=response_draws,
            pair_overlap=jnp.asarray(pair_leaves["pair_overlap"]),
        ),
    )
    flat = np.reshape(np.asarray(response_draws), (NUM_DRAWS, -1))
    moment = np.asarray(model.power_moment(OFF_FIDUCIAL))
    new = np.tile(moment, (NUM_PAIRS, NUM_PAIRS)) * (flat.T @ flat / NUM_DRAWS)

    np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-12 * np.abs(old).max())


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def test_network_from_pairs_keeps_only_the_counted_entries(
    network: Network, pair_leaves: dict[str, np.ndarray]
) -> None:
    assert network.scale.shape == (int(pair_leaves["mask"].sum()),)
    assert network.response_moment is not None
    assert network.response_moment.shape == (network.scale.size,) * 2
    assert bool(jnp.all(jnp.isfinite(network.scale)))


# --------------------------------------------------------------------------- #
# DenseGaussianLikelihood
# --------------------------------------------------------------------------- #
def test_log_likelihood_is_the_multivariate_normal_with_the_catalog_covariance(
    likelihood: DenseGaussianLikelihood, network: Network, observed: jax.Array
) -> None:
    (prediction, factor) = likelihood.evaluate(OFF_FIDUCIAL)
    assert prediction.power_moment is not None and network.response_moment is not None
    covariance = prediction.power_moment[network.bins][
        :, network.bins
    ] * network.response_moment + jnp.diag(network.scale**2)

    np.testing.assert_allclose(
        likelihood.log_likelihood(OFF_FIDUCIAL)[0],
        dist.MultivariateNormal(factor.mean, covariance).log_prob(observed),
        rtol=1e-10,
    )


def test_log_likelihood_without_shot_noise_is_the_independent_normal(
    likelihood: DenseGaussianLikelihood, network: Network, observed: jax.Array
) -> None:
    diagonal = DenseGaussianLikelihood(
        likelihood.model, network, observed, shot_noise=False
    )
    mean = network.pair_overlap * likelihood.model.spectrum(OFF_FIDUCIAL)[network.bins]

    np.testing.assert_allclose(
        diagonal.log_likelihood(OFF_FIDUCIAL)[0],
        jnp.sum(dist.Normal(mean, network.scale).log_prob(observed), axis=-1),
        rtol=1e-12,
    )


def test_log_likelihood_scores_each_realization_against_one_factor(
    likelihood: DenseGaussianLikelihood, observed: jax.Array
) -> None:
    together = likelihood.log_likelihood(OFF_FIDUCIAL)[0]
    alone = [
        DenseGaussianLikelihood(
            likelihood.model, likelihood.network, row
        ).log_likelihood(OFF_FIDUCIAL)[0]
        for row in observed
    ]

    assert together.shape == (observed.shape[0],)
    np.testing.assert_allclose(together, jnp.stack(alone), rtol=1e-12)


def test_log_likelihood_ignores_the_data_at_masked_entries(
    model: CatalogModel,
    network: Network,
    pair_leaves: dict[str, np.ndarray],
) -> None:
    data = np.ones((NUM_PAIRS, NUM_BINS)) * float(jnp.max(model.spectrum(FIDUCIALS)))
    perturbed = np.where(pair_leaves["mask"], data, 1e30)

    np.testing.assert_array_equal(
        DenseGaussianLikelihood(model, network, network.compress(data)).log_likelihood(
            FIDUCIALS
        )[0],
        DenseGaussianLikelihood(
            model, network, network.compress(perturbed)
        ).log_likelihood(FIDUCIALS)[0],
    )


def test_log_likelihood_with_shot_noise_and_no_response_moment_raises(
    model: CatalogModel, pair_leaves: dict[str, np.ndarray]
) -> None:
    network = Network.from_pairs(**pair_leaves)

    with pytest.raises(ValueError, match="response_moment"):
        DenseGaussianLikelihood(
            model, network, jnp.zeros(network.scale.shape)
        ).log_likelihood(FIDUCIALS)


# --------------------------------------------------------------------------- #
# pair_response_moment
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def triangle() -> list[Any]:
    return [load_detector(name) for name in ("E1", "E2", "E3")]


RESPONSE_FREQUENCIES = np.array([5.0, 30.0, 300.0])


def test_pair_response_moment_does_not_depend_on_the_chunk_size(
    triangle: list[Any],
) -> None:
    key = jax.random.key(4)
    whole = pair_response_moment(key, 50, RESPONSE_FREQUENCIES, triangle, chunk_size=64)
    chunked = pair_response_moment(
        key, 50, RESPONSE_FREQUENCIES, triangle, chunk_size=7
    )

    np.testing.assert_allclose(chunked, whole, rtol=1e-12, atol=1e-15)


def test_pair_response_moment_is_symmetric_positive_semidefinite(
    triangle: list[Any],
) -> None:
    moment = pair_response_moment(
        jax.random.key(5), 200, RESPONSE_FREQUENCIES, triangle
    )
    matrix = np.reshape(np.asarray(moment), (3 * RESPONSE_FREQUENCIES.size,) * 2)
    eigenvalues = np.linalg.eigvalsh(matrix)

    assert moment.shape == (3, RESPONSE_FREQUENCIES.size) * 2
    np.testing.assert_array_equal(matrix, matrix.T)
    assert eigenvalues.min() >= -1e-12 * eigenvalues.max()


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
GRIDS = {"H0": jnp.array([60.0, 67.66, 75.0]), "Omega_m": jnp.array([0.28, 0.32])}


def test_grid_log_likelihood_matches_the_pointwise_log_likelihood(
    likelihood: DenseGaussianLikelihood, observed: jax.Array
) -> None:
    grid = grid_log_likelihood(likelihood, GRIDS, fixed={"xi_0": 1.0})
    pointwise = [
        [
            likelihood.log_likelihood(
                {**FIDUCIALS, "H0": h0, "Omega_m": omega_m, "xi_0": 1.0}
            )[0]
            for omega_m in GRIDS["Omega_m"]
        ]
        for h0 in GRIDS["H0"]
    ]

    assert grid.shape == (3, 2, observed.shape[0])
    np.testing.assert_allclose(grid, jnp.asarray(pointwise), rtol=1e-10)


def test_grid_log_likelihood_compiles_once_for_data_of_the_same_shape(
    likelihood: DenseGaussianLikelihood,
    observed: jax.Array,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces: list[None] = []
    log_likelihood = DenseGaussianLikelihood.log_likelihood

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        traces.append(None)
        return log_likelihood(self, *args, **kwargs)

    monkeypatch.setattr(DenseGaussianLikelihood, "log_likelihood", counted)
    # A grid length no other test uses, so the first call compiles here.
    grids = {"H0": jnp.linspace(62.0, 72.0, 5)}
    results = [
        grid_log_likelihood(
            DenseGaussianLikelihood(likelihood.model, likelihood.network, data), grids
        )
        for data in (observed, 2.0 * observed)
    ]

    assert len(traces) == 1
    assert not np.allclose(results[0], results[1], rtol=1e-6, atol=0.0)


def test_grid_log_prior_outside_the_support_is_negative_infinite() -> None:
    priors = {
        "H0": dist.Uniform(65.0, 80.0),
        "Omega_m": dist.Normal(0.3, 0.01),
    }

    with pytest.warns(UserWarning, match="Out-of-support"):
        prior = grid_log_prior(priors, GRIDS)

    assert prior.shape == (3, 2)
    assert bool(jnp.all(jnp.isneginf(prior[0])))
    assert bool(jnp.all(jnp.isfinite(prior[1:])))


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def prediction(model: CatalogModel) -> Prediction:
    return model.predict(OFF_FIDUCIAL)


def test_shape_correction_and_shot_variance_give_the_amplitude_information(
    likelihood: DenseGaussianLikelihood, prediction: Prediction, network: Network
) -> None:
    r""":math:`\mu^T C^{-1} \mu = \rho^2 / (1 + \rho^2 \sigma^2 - \delta)` (App. B)."""
    _, factor = likelihood.evaluate(OFF_FIDUCIAL)
    whitened = solve_triangular(factor.cholesky, factor.mean, lower=True)
    _, squared_snr = amplitude_template(prediction, network)
    shot = shot_noise_variance(prediction, network)
    delta = shape_correction(prediction, network)

    assert float(whitened @ whitened) == pytest.approx(
        float(squared_snr / (1.0 + squared_snr * shot - delta)), rel=1e-9
    )


def test_shape_correction_is_bounded_by_the_largest_orthogonal_eigenvalue(
    prediction: Prediction, network: Network
) -> None:
    _, squared_snr = amplitude_template(prediction, network)
    shot = float(squared_snr * shot_noise_variance(prediction, network))
    largest = float(
        jnp.max(
            jnp.linalg.eigvalsh(
                whitened_catalog_covariance(prediction, network, orthogonal=True)
            )
        )
    )
    delta = float(shape_correction(prediction, network))

    assert 0.0 <= delta <= shot * largest / (1.0 + largest) * (1.0 + 1e-12)


def test_orthogonal_whitened_covariance_annihilates_the_mean_direction(
    prediction: Prediction, network: Network
) -> None:
    direction = network.pair_overlap * prediction.mean[network.bins] / network.scale
    projected = whitened_catalog_covariance(prediction, network, orthogonal=True)
    full = whitened_catalog_covariance(prediction, network)

    assert float(jnp.linalg.norm(projected @ direction)) <= 1e-12 * float(
        jnp.linalg.norm(full @ direction)
    )


def test_amplitude_cumulants_shot_variance_is_the_response_moment_variance(
    model: CatalogModel,
    prediction: Prediction,
    network: Network,
    response_draws: jax.Array,
) -> None:
    """The per-draw catalog sum and the held moment of the same draws agree."""
    cumulants = amplitude_cumulants(model, OFF_FIDUCIAL, network, response_draws)

    assert float(cumulants.shot_variance) == pytest.approx(
        float(shot_noise_variance(prediction, network)), rel=1e-10
    )


def test_amplitude_cumulants_of_a_positive_template_are_right_skewed(
    model: CatalogModel, pair_leaves: dict[str, np.ndarray], response_draws: jax.Array
) -> None:
    """Positive overlaps and responses make every source's contribution positive."""
    leaves = {**pair_leaves, "pair_overlap": np.abs(pair_leaves["pair_overlap"])}
    network = Network.from_pairs(**leaves)

    cumulants = amplitude_cumulants(model, OFF_FIDUCIAL, network, response_draws)

    assert float(cumulants.third_cumulant) > 0.0
    assert float(cumulants.skewness) > 0.0


# --------------------------------------------------------------------------- #
# The dense Fisher matrix
# --------------------------------------------------------------------------- #
FISHER_PARAMETERS = ("H0", "Omega_m")


def test_dense_fisher_without_shot_noise_is_the_per_bin_fisher_sum(
    model: CatalogModel, pair_leaves: dict[str, np.ndarray]
) -> None:
    """One pair of unit overlap is the combined spectrum the per-bin matrix scores."""
    scale = pair_leaves["scale"][0]
    network = Network.from_pairs(scale[None], np.ones((1, NUM_BINS)))
    likelihood = DenseGaussianLikelihood(
        model, network, jnp.zeros(NUM_BINS), shot_noise=False
    )

    fisher = dense_fisher_matrix(likelihood, OFF_FIDUCIAL, FISHER_PARAMETERS)
    per_bin = fisher_matrix_per_bin(
        model.spectrum, OFF_FIDUCIAL, FISHER_PARAMETERS, scale=jnp.asarray(scale)
    )

    np.testing.assert_allclose(fisher.total, per_bin.sum(axis=0), rtol=1e-10)
    np.testing.assert_array_equal(fisher.covariance, 0.0)


def test_dense_fisher_covariance_term_is_symmetric_positive_semidefinite(
    likelihood: DenseGaussianLikelihood,
) -> None:
    covariance = np.asarray(
        dense_fisher_matrix(likelihood, OFF_FIDUCIAL, FISHER_PARAMETERS).covariance
    )

    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12 * np.abs(covariance).max()
    assert np.abs(covariance).max() > 0.0
