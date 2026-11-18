"""The catalog model, the dense Gaussian likelihood and what is derived from it.

The catalog-bound tests use the closed-form mock reference catalog of
``astrogwb_mock_population`` on a 12-bin grid and two synthetic detector pairs
whose response draws vary across the band. The constructor tests use the
Einstein Telescope triangle.
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
    build_reference_catalog,
    build_reference_spectrum,
    mock_merger_rate,
)
from jax.scipy.linalg import solve_triangular
from jax.tree_util import Partial
from jax.typing import ArrayLike

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import (
    connected_overlap,
    load_detector,
    pair_names,
    pair_response,
)
from astrogwb.detector.response import _isotropic_angles
from astrogwb.importance.diagnostics import amplitude_cumulants
from astrogwb.importance.model import CatalogModel, interpolation_weights
from astrogwb.importance.spectrum import (
    INCLINATION_MEAN,
    INCLINATION_RAW_SECOND_MOMENT,
)
from astrogwb.inference import (
    DenseGaussianLikelihood,
    SpectralDensityPrediction,
    amplitude_template,
    grid_log_likelihood,
    grid_log_prior,
    shape_correction,
    shot_noise_variance,
    whitened_catalog_covariance,
)
from astrogwb.inference.gaussian import _Network
from astrogwb.utils import gauss_legendre_nodes_weights

NUM_PAIRS = 2
NUM_BINS = 12
NUM_DRAWS = 64
OBSERVATION_TIME = 2.0
OFF_FIDUCIAL = {**FIDUCIALS, "H0": 70.0}
PAIRS = (("A", "B"), ("A", "C"))

#: Converts a sampled response moment, inclination included, into the
#: connected overlap, whose inclination scatter the second moment carries.
RESPONSE_TO_CONNECTED = INCLINATION_MEAN**2 / INCLINATION_RAW_SECOND_MOMENT


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def catalog() -> Any:
    return build_reference_catalog(num_sources=16)


@pytest.fixture(scope="module")
def frequencies(catalog: Any) -> np.ndarray:
    return np.asarray(catalog.frequencies)[::4][:NUM_BINS]


@pytest.fixture(scope="module")
def model_factory(catalog: Any, frequencies: np.ndarray) -> Callable[..., CatalogModel]:
    def build(reweight_masses: bool = False) -> CatalogModel:
        return build_reference_spectrum(
            catalog,
            merger_rate=mock_merger_rate,
            frequencies=frequencies,
            reweight_masses=reweight_masses,
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
def keyed_draws(response_draws: jax.Array) -> dict[tuple[str, str], jax.Array]:
    return {pair: response_draws[:, index] for index, pair in enumerate(PAIRS)}


@pytest.fixture(scope="module")
def pair_leaves(model: CatalogModel) -> dict[str, Any]:
    """Scale, ORF and mask of two pairs; the second pair drops every third bin."""
    typical = float(jnp.max(model.spectral_density(FIDUCIALS)))
    rng = np.random.default_rng(0)
    mask = np.ones((NUM_PAIRS, NUM_BINS), dtype=bool)
    mask[1, ::3] = False
    return {
        "scale": rng.uniform(0.05, 0.2, (NUM_PAIRS, NUM_BINS)) * typical,
        "pair_overlap": np.stack(
            [np.linspace(0.9, 0.4, NUM_BINS), np.linspace(-0.3, 0.2, NUM_BINS)]
        ),
        "mask": mask,
        "observation_time": OBSERVATION_TIME,
        "pairs": PAIRS,
    }


@pytest.fixture(scope="module")
def network(pair_leaves: dict[str, Any], response_draws: jax.Array) -> _Network:
    flat = np.reshape(np.asarray(response_draws), (NUM_DRAWS, -1))
    moment = (flat.T @ flat / NUM_DRAWS).reshape((NUM_PAIRS, NUM_BINS) * 2)
    return _Network.from_leaves(
        **pair_leaves, connected_overlap=RESPONSE_TO_CONNECTED * moment
    )


@pytest.fixture(scope="module")
def observed(
    model: CatalogModel, pair_leaves: dict[str, Any], network: _Network
) -> jax.Array:
    """Three realizations about the fiducial mean, on the counted entries, ``(3, D)``."""
    mean = pair_leaves["pair_overlap"] * np.asarray(model.spectral_density(FIDUCIALS))
    noise = np.random.default_rng(1).normal(size=(3, NUM_PAIRS, NUM_BINS))
    return network.compress(mean * 1.05 + noise * pair_leaves["scale"])


@pytest.fixture(scope="module")
def likelihood(
    model: CatalogModel, network: _Network, observed: jax.Array
) -> DenseGaussianLikelihood[CatalogModel]:
    return DenseGaussianLikelihood._from_network(model, network, observed)


# --------------------------------------------------------------------------- #
# interpolation_weights and the intensity second moment
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


def test_second_moment_held_node_moments_match_the_weighted_gather_at_the_fiducials(
    model_factory: Callable[..., CatalogModel],
) -> None:
    """At the fiducials every weight is one, so both paths are the same moment."""
    held = model_factory()
    weighted = model_factory(reweight_masses=True)
    assert held.node_moments is not None and weighted.node_moments is None

    np.testing.assert_allclose(
        weighted.intensity_second_moment(FIDUCIALS),
        held.intensity_second_moment(FIDUCIALS),
        rtol=1e-10,
    )


def test_second_moment_is_symmetric_positive_semidefinite(model: CatalogModel) -> None:
    moment = np.asarray(model.intensity_second_moment(OFF_FIDUCIAL))
    eigenvalues = np.linalg.eigvalsh(moment)

    np.testing.assert_allclose(moment, moment.T, rtol=1e-14)
    assert eigenvalues.min() >= -1e-10 * eigenvalues.max()


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
def test_network_from_leaves_keeps_only_the_counted_entries(
    network: _Network, pair_leaves: dict[str, Any]
) -> None:
    assert network.scale.shape == (int(pair_leaves["mask"].sum()),)
    assert network.connected_overlap is not None
    assert network.connected_overlap.shape == (network.scale.size,) * 2


def test_network_from_leaves_drops_entries_with_infinite_noise(
    pair_leaves: dict[str, Any],
) -> None:
    scale = np.array(pair_leaves["scale"])
    scale[0, :3] = np.inf
    leaves: dict[str, Any] = {**pair_leaves, "scale": scale, "mask": None}
    network = _Network.from_leaves(**leaves)

    assert network.scale.shape == (NUM_PAIRS * NUM_BINS - 3,)


# --------------------------------------------------------------------------- #
# DenseGaussianLikelihood
# --------------------------------------------------------------------------- #
def test_log_likelihood_is_the_multivariate_normal_with_the_catalog_covariance(
    likelihood: DenseGaussianLikelihood[CatalogModel],
    network: _Network,
    observed: jax.Array,
) -> None:
    (prediction, factor) = likelihood.evaluate(OFF_FIDUCIAL)
    moment = prediction.intensity_second_moment
    assert moment is not None and network.connected_overlap is not None
    seconds = OBSERVATION_TIME * SECONDS_PER_YEAR
    covariance = moment[network.bins][
        :, network.bins
    ] * network.connected_overlap / seconds + jnp.diag(network.scale**2)

    np.testing.assert_allclose(
        likelihood.log_likelihood(OFF_FIDUCIAL)[0],
        dist.MultivariateNormal(factor.mean, covariance).log_prob(observed),
        rtol=1e-10,
    )


def test_log_likelihood_without_shot_noise_is_the_independent_normal(
    likelihood: DenseGaussianLikelihood[CatalogModel],
    network: _Network,
    observed: jax.Array,
) -> None:
    diagonal = DenseGaussianLikelihood._from_network(
        likelihood.model, network, observed, shot_noise=False
    )
    mean = (
        network.pair_overlap
        * likelihood.model.spectral_density(OFF_FIDUCIAL)[network.bins]
    )

    np.testing.assert_allclose(
        diagonal.log_likelihood(OFF_FIDUCIAL)[0],
        jnp.sum(dist.Normal(mean, network.scale).log_prob(observed), axis=-1),
        rtol=1e-12,
    )


def test_log_likelihood_scores_each_realization_against_one_factor(
    likelihood: DenseGaussianLikelihood[CatalogModel],
    network: _Network,
    observed: jax.Array,
) -> None:
    together = likelihood.log_likelihood(OFF_FIDUCIAL)[0]
    alone = [
        DenseGaussianLikelihood._from_network(
            likelihood.model, network, row
        ).log_likelihood(OFF_FIDUCIAL)[0]
        for row in observed
    ]

    assert together.shape == (observed.shape[0],)
    np.testing.assert_allclose(together, jnp.stack(alone), rtol=1e-12)


def test_log_likelihood_ignores_the_data_at_masked_entries(
    model: CatalogModel, network: _Network, pair_leaves: dict[str, Any]
) -> None:
    data = np.ones((NUM_PAIRS, NUM_BINS)) * float(
        jnp.max(model.spectral_density(FIDUCIALS))
    )
    perturbed = np.where(pair_leaves["mask"], data, 1e30)

    np.testing.assert_array_equal(
        DenseGaussianLikelihood._from_network(
            model, network, network.compress(data)
        ).log_likelihood(FIDUCIALS)[0],
        DenseGaussianLikelihood._from_network(
            model, network, network.compress(perturbed)
        ).log_likelihood(FIDUCIALS)[0],
    )


def test_log_likelihood_with_shot_noise_and_no_connected_overlap_raises(
    model: CatalogModel, pair_leaves: dict[str, Any]
) -> None:
    network = _Network.from_leaves(**pair_leaves)
    likelihood = DenseGaussianLikelihood._from_network(
        model, network, jnp.zeros(network.scale.shape)
    )

    with pytest.raises(ValueError, match="connected overlap"):
        likelihood.log_likelihood(FIDUCIALS)


def test_log_likelihood_without_data_raises(
    model: CatalogModel, network: _Network
) -> None:
    likelihood = DenseGaussianLikelihood._from_network(model, network, None)

    with pytest.raises(ValueError, match="without observed data"):
        likelihood.log_likelihood(FIDUCIALS)


def test_log_likelihood_of_a_model_on_another_grid_raises(
    catalog: Any, network: _Network, observed: jax.Array
) -> None:
    other = build_reference_spectrum(
        catalog,
        merger_rate=mock_merger_rate,
        frequencies=np.asarray(catalog.frequencies)[: NUM_BINS + 1],
        reweight_masses=False,
    )
    likelihood = DenseGaussianLikelihood._from_network(other, network, observed)

    with pytest.raises(ValueError, match="frequency grid"):
        likelihood.log_likelihood(FIDUCIALS)


# --------------------------------------------------------------------------- #
# The public constructor, on the ET triangle
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def triangle() -> list[Any]:
    return [load_detector(name) for name in ("E1", "E2", "E3")]


@pytest.fixture(scope="module")
def triangle_data(
    model: CatalogModel, frequencies: np.ndarray, triangle: list[Any]
) -> dict[tuple[str, str], np.ndarray]:
    """One realization per sorted pair, a little above the fiducial spectrum."""
    spectrum = np.asarray(model.spectral_density(FIDUCIALS))
    rng = np.random.default_rng(2)
    return {
        pair: spectrum * (1.0 + 0.1 * rng.normal(size=frequencies.size))
        for pair in pair_names(triangle)
    }


def _triangle_likelihood(
    model: CatalogModel,
    detectors: list[Any],
    frequencies: np.ndarray,
    observed: Mapping[tuple[str, str], ArrayLike] | None,
    **settings: Any,
) -> DenseGaussianLikelihood[CatalogModel]:
    settings.setdefault("shot_noise", False)
    return DenseGaussianLikelihood(
        model,
        detectors,
        frequencies=frequencies,
        observation_time=OBSERVATION_TIME,
        observed=observed,
        **settings,
    )


def test_constructor_does_not_depend_on_the_detector_order(
    model: CatalogModel,
    frequencies: np.ndarray,
    triangle: list[Any],
    triangle_data: dict[tuple[str, str], np.ndarray],
) -> None:
    forward = _triangle_likelihood(model, triangle, frequencies, triangle_data)
    backward = _triangle_likelihood(model, triangle[::-1], frequencies, triangle_data)

    np.testing.assert_allclose(
        forward.log_likelihood(OFF_FIDUCIAL)[0],
        backward.log_likelihood(OFF_FIDUCIAL)[0],
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "rekey",
    [
        lambda data: dict(list(data.items())[1:]),
        lambda data: {**data, ("E1", "E4"): next(iter(data.values()))},
        lambda data: {(b, a): value for (a, b), value in data.items()},
    ],
    ids=["missing", "extra", "unsorted"],
)
def test_constructor_with_mismatched_pair_keys_raises(
    model: CatalogModel,
    frequencies: np.ndarray,
    triangle: list[Any],
    triangle_data: dict[tuple[str, str], np.ndarray],
    rekey: Callable[[dict[Any, Any]], dict[Any, Any]],
) -> None:
    with pytest.raises(ValueError, match="sorted detector pair"):
        _triangle_likelihood(model, triangle, frequencies, rekey(triangle_data))


def test_constructor_mask_drops_the_masked_bins(
    model: CatalogModel,
    frequencies: np.ndarray,
    triangle: list[Any],
    triangle_data: dict[tuple[str, str], np.ndarray],
) -> None:
    mask = {pair: np.arange(frequencies.size) % 2 == 0 for pair in triangle_data}
    full = _triangle_likelihood(model, triangle, frequencies, triangle_data)
    masked = _triangle_likelihood(
        model, triangle, frequencies, triangle_data, mask=mask
    )

    assert masked.observed is not None and full.observed is not None
    assert masked.observed.shape[-1] < full.observed.shape[-1]


def test_constructor_with_shot_noise_gives_a_finite_log_likelihood(
    model: CatalogModel,
    frequencies: np.ndarray,
    triangle: list[Any],
    triangle_data: dict[tuple[str, str], np.ndarray],
) -> None:
    likelihood = _triangle_likelihood(
        model, triangle, frequencies, triangle_data, shot_noise=True, num_draws=256
    )

    assert bool(jnp.isfinite(likelihood.log_likelihood(OFF_FIDUCIAL)[0]))


# --------------------------------------------------------------------------- #
# connected_overlap
# --------------------------------------------------------------------------- #
RESPONSE_FREQUENCIES = np.array([5.0, 30.0, 300.0])


def test_connected_overlap_does_not_depend_on_the_chunk_size(
    triangle: list[Any],
) -> None:
    key = jax.random.key(4)
    whole = connected_overlap(key, 50, RESPONSE_FREQUENCIES, triangle, chunk_size=64)
    chunked = connected_overlap(key, 50, RESPONSE_FREQUENCIES, triangle, chunk_size=7)

    np.testing.assert_allclose(chunked, whole, rtol=1e-12, atol=1e-15)


def test_connected_overlap_is_symmetric_positive_semidefinite(
    triangle: list[Any],
) -> None:
    overlap = connected_overlap(jax.random.key(5), 200, RESPONSE_FREQUENCIES, triangle)
    matrix = np.reshape(np.asarray(overlap), (3 * RESPONSE_FREQUENCIES.size,) * 2)
    eigenvalues = np.linalg.eigvalsh(matrix)

    assert overlap.shape == (3, RESPONSE_FREQUENCIES.size) * 2
    np.testing.assert_array_equal(matrix, matrix.T)
    assert eigenvalues.min() >= -1e-12 * eigenvalues.max()


def test_connected_overlap_is_the_inclination_averaged_response_moment(
    triangle: list[Any],
) -> None:
    """The per-source ORF moment, inclination integrated at the same sky draws.

    The inclination scatter :math:`c_\\iota / (2/5)^2` the moment carries is
    the second moment's, so it is divided out.
    """
    key, num_draws = jax.random.key(6), 40
    right_ascension, declination, polarization, _ = _isotropic_angles(key, num_draws)
    nodes, weights = gauss_legendre_nodes_weights(-1.0, 1.0, 8)
    responses = jax.vmap(
        lambda cosine: pair_response(
            RESPONSE_FREQUENCIES,
            triangle,
            right_ascension,
            declination,
            polarization,
            cosine,
        )
    )(jnp.asarray(nodes))
    flat = jnp.reshape(responses, (len(nodes), num_draws, -1))
    moment = jnp.einsum("n,nli,nlj->ij", jnp.asarray(weights) / 2.0, flat, flat)

    np.testing.assert_allclose(
        np.reshape(
            np.asarray(
                connected_overlap(key, num_draws, RESPONSE_FREQUENCIES, triangle)
            ),
            moment.shape,
        ),
        RESPONSE_TO_CONNECTED * moment / num_draws,
        rtol=1e-10,
        atol=1e-14,
    )


def test_connected_overlap_paper_normalization_is_four_twenty_fifths(
    triangle: list[Any],
) -> None:
    key = jax.random.key(7)
    normalized = connected_overlap(key, 30, RESPONSE_FREQUENCIES, triangle)
    paper = connected_overlap(key, 30, RESPONSE_FREQUENCIES, triangle, normalize=False)

    np.testing.assert_allclose(paper, 4.0 / 25.0 * normalized, rtol=1e-14)


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
GRIDS = {"H0": jnp.array([60.0, 67.66, 75.0]), "Omega_m": jnp.array([0.28, 0.32])}


def _delegate(
    model: CatalogModel, params: Mapping[str, ArrayLike], *, shot_noise: bool
) -> SpectralDensityPrediction:
    return model(params, shot_noise=shot_noise)


def test_grid_log_likelihood_matches_the_pointwise_log_likelihood(
    likelihood: DenseGaussianLikelihood[CatalogModel], observed: jax.Array
) -> None:
    fixed = {name: value for name, value in FIDUCIALS.items() if name not in GRIDS}
    grid = grid_log_likelihood(likelihood, GRIDS, fixed=fixed)
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


def test_grid_log_likelihood_of_a_partial_model_matches_the_model(
    likelihood: DenseGaussianLikelihood[CatalogModel],
    network: _Network,
    observed: jax.Array,
) -> None:
    partial = DenseGaussianLikelihood._from_network(
        Partial(_delegate, likelihood.model), network, observed
    )
    fixed = {name: value for name, value in FIDUCIALS.items() if name not in GRIDS}

    np.testing.assert_allclose(
        grid_log_likelihood(partial, GRIDS, fixed=fixed),
        grid_log_likelihood(likelihood, GRIDS, fixed=fixed),
        rtol=1e-12,
    )


def test_grid_log_likelihood_compiles_once_for_data_of_the_same_shape(
    likelihood: DenseGaussianLikelihood[CatalogModel],
    network: _Network,
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
            DenseGaussianLikelihood._from_network(likelihood.model, network, data),
            grids,
            fixed={
                name: value for name, value in FIDUCIALS.items() if name not in grids
            },
        )
        for data in (observed, 2.0 * observed)
    ]

    assert len(traces) == 1
    assert not np.allclose(results[0], results[1], rtol=1e-6, atol=0.0)


def test_grid_log_prior_outside_the_support_is_negative_infinite() -> None:
    priors = {
        # Explicit: numpyro only validates by default from 0.22.
        "H0": dist.Uniform(65.0, 80.0, validate_args=True),
        "Omega_m": dist.Normal(0.3, 0.01, validate_args=True),
    }

    with pytest.warns(UserWarning, match="Out-of-support"):
        prior = grid_log_prior(priors, GRIDS)

    assert prior.shape == (3, 2)
    assert bool(jnp.all(jnp.isneginf(prior[0])))
    assert bool(jnp.all(jnp.isfinite(prior[1:])))


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_shape_correction_and_shot_variance_give_the_amplitude_information(
    likelihood: DenseGaussianLikelihood[CatalogModel],
) -> None:
    r""":math:`\mu^T C^{-1} \mu = \rho^2 / (1 + \rho^2 \sigma^2 - \delta)` (App. B)."""
    _, factor = likelihood.evaluate(OFF_FIDUCIAL)
    whitened = solve_triangular(factor.cholesky, factor.mean, lower=True)
    _, squared_snr = amplitude_template(likelihood, OFF_FIDUCIAL)
    shot = shot_noise_variance(likelihood, OFF_FIDUCIAL)
    delta = shape_correction(likelihood, OFF_FIDUCIAL)

    assert float(whitened @ whitened) == pytest.approx(
        float(squared_snr / (1.0 + squared_snr * shot - delta)), rel=1e-9
    )


def test_shape_correction_is_bounded_by_the_largest_orthogonal_eigenvalue(
    likelihood: DenseGaussianLikelihood[CatalogModel],
) -> None:
    _, squared_snr = amplitude_template(likelihood, OFF_FIDUCIAL)
    shot = float(squared_snr * shot_noise_variance(likelihood, OFF_FIDUCIAL))
    largest = float(
        jnp.max(
            jnp.linalg.eigvalsh(
                whitened_catalog_covariance(likelihood, OFF_FIDUCIAL, orthogonal=True)
            )
        )
    )
    delta = float(shape_correction(likelihood, OFF_FIDUCIAL))

    assert 0.0 <= delta <= shot * largest / (1.0 + largest) * (1.0 + 1e-12)


def test_orthogonal_whitened_covariance_annihilates_the_mean_direction(
    likelihood: DenseGaussianLikelihood[CatalogModel], network: _Network
) -> None:
    mean = likelihood.model.spectral_density(OFF_FIDUCIAL)[network.bins]
    direction = network.pair_overlap * mean / network.scale
    projected = whitened_catalog_covariance(likelihood, OFF_FIDUCIAL, orthogonal=True)
    full = whitened_catalog_covariance(likelihood, OFF_FIDUCIAL)

    assert float(jnp.linalg.norm(projected @ direction)) <= 1e-12 * float(
        jnp.linalg.norm(full @ direction)
    )


def test_amplitude_cumulants_shot_variance_is_the_response_moment_variance(
    likelihood: DenseGaussianLikelihood[CatalogModel],
    keyed_draws: dict[tuple[str, str], jax.Array],
) -> None:
    """The per-draw catalog sum and the connected overlap of the same draws agree."""
    cumulants = amplitude_cumulants(likelihood, OFF_FIDUCIAL, keyed_draws)

    assert float(cumulants.shot_variance) == pytest.approx(
        float(shot_noise_variance(likelihood, OFF_FIDUCIAL)), rel=1e-10
    )


def test_amplitude_cumulants_of_a_positive_template_are_right_skewed(
    model: CatalogModel,
    pair_leaves: dict[str, Any],
    keyed_draws: dict[tuple[str, str], jax.Array],
) -> None:
    """Positive overlaps and responses make every source's contribution positive."""
    leaves = {**pair_leaves, "pair_overlap": np.abs(pair_leaves["pair_overlap"])}
    likelihood = DenseGaussianLikelihood._from_network(
        model, _Network.from_leaves(**leaves), None, shot_noise=False
    )

    cumulants = amplitude_cumulants(likelihood, OFF_FIDUCIAL, keyed_draws)

    assert float(cumulants.third_cumulant) > 0.0
    assert float(cumulants.skewness) > 0.0
