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
from jax.tree_util import Partial
from jax.typing import ArrayLike

from astrogwb.background.importance.model import CatalogModel, interpolation_weights
from astrogwb.background.importance.spectrum import (
    INCLINATION_MEAN,
    INCLINATION_RAW_SECOND_MOMENT,
)
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import (
    NORMALIZED_ORF_SCALE,
    generalized_overlap_reduction_function,
    load_detector,
    pair_names,
    pair_response,
)
from astrogwb.detector.response import _isotropic_angles
from astrogwb.inference import (
    DenseGaussianLikelihood,
    SpectralDensityPrediction,
    grid_log_likelihood,
    grid_log_prior,
)
from astrogwb.inference.gaussian import _Network
from astrogwb.statistics import (
    amplitude,
    amplitude_variance,
    amplitude_variance_budget,
    amplitude_weights,
)
from astrogwb.utils import gauss_legendre_nodes_weights
from astrogwb.waveform.intensity import QUADRUPOLE_STOKES_WEIGHTS, StokesPair

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
    assert isinstance(network.connected_overlap, jax.Array)
    assert network.connected_overlap.shape == (network.scale.size,) * 2


def test_network_from_leaves_with_both_overlap_forms_raises(
    pair_leaves: dict[str, Any],
) -> None:
    overlap = np.zeros((NUM_PAIRS, NUM_BINS) * 2)

    with pytest.raises(ValueError, match="not both"):
        _Network.from_leaves(
            **pair_leaves,
            connected_overlap=overlap,
            generalized_orfs={"II": overlap},
        )


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
    assert moment is not None
    assert isinstance(network.connected_overlap, jax.Array)
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


def test_constructor_covariance_does_not_depend_on_keeping_the_generalized_orfs(
    model: CatalogModel,
    frequencies: np.ndarray,
    triangle: list[Any],
) -> None:
    combined, kept = (
        _triangle_likelihood(
            model,
            triangle,
            frequencies,
            None,
            shot_noise=True,
            num_draws=256,
            keep_generalized_orfs=keep,
        ).evaluate(OFF_FIDUCIAL)[1]
        for keep in (False, True)
    )

    np.testing.assert_allclose(kept.cholesky, combined.cholesky, rtol=1e-12)


def test_constructor_keeping_generalized_orfs_splits_its_covariance(
    model: CatalogModel,
    frequencies: np.ndarray,
    triangle: list[Any],
) -> None:
    likelihood = _triangle_likelihood(
        model,
        triangle,
        frequencies,
        None,
        shot_noise=True,
        num_draws=256,
        keep_generalized_orfs=True,
    )
    _, factor = likelihood.evaluate(OFF_FIDUCIAL)
    components = likelihood.covariance_components(OFF_FIDUCIAL)

    np.testing.assert_allclose(
        sum(components.values()), factor.cholesky @ factor.cholesky.T, rtol=1e-10
    )


# --------------------------------------------------------------------------- #
# generalized_overlap_reduction_function
# --------------------------------------------------------------------------- #
RESPONSE_FREQUENCIES = np.array([5.0, 30.0, 300.0])

#: Uniform polarization angles on [0, pi), exact for the products of two pair
#: responses, whose harmonics stop at 8 psi.
NUM_POLARIZATION_NODES = 8


def test_generalized_orfs_do_not_depend_on_the_chunk_size(
    triangle: list[Any],
) -> None:
    key = jax.random.key(4)
    whole = generalized_overlap_reduction_function(
        key, 50, RESPONSE_FREQUENCIES, triangle, chunk_size=64
    )
    chunked = generalized_overlap_reduction_function(
        key, 50, RESPONSE_FREQUENCIES, triangle, chunk_size=7
    )

    for pair, values in whole.items():
        np.testing.assert_allclose(chunked[pair], values, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("pair", ["II", "QQ", "UU", "VV"])
def test_generalized_orf_diagonal_component_is_symmetric_positive_semidefinite(
    triangle: list[Any], pair: StokesPair
) -> None:
    orfs = generalized_overlap_reduction_function(
        jax.random.key(5), 200, RESPONSE_FREQUENCIES, triangle
    )
    matrix = np.reshape(np.asarray(orfs[pair]), (3 * RESPONSE_FREQUENCIES.size,) * 2)
    eigenvalues = np.linalg.eigvalsh(matrix)

    assert orfs[pair].shape == (3, RESPONSE_FREQUENCIES.size) * 2
    np.testing.assert_array_equal(matrix, matrix.T)
    assert eigenvalues.min() >= -1e-12 * eigenvalues.max()


def test_generalized_orf_vi_is_iv_with_the_pairs_swapped(
    triangle: list[Any],
) -> None:
    orfs = generalized_overlap_reduction_function(
        jax.random.key(6), 50, RESPONSE_FREQUENCIES, triangle, components=("IV", "VI")
    )

    np.testing.assert_array_equal(orfs["VI"], jnp.transpose(orfs["IV"], (2, 3, 0, 1)))


def test_quadrupole_combination_is_the_source_averaged_response_moment(
    triangle: list[Any],
) -> None:
    r"""The per-source ORF moment, inclination and polarization integrated exactly.

    At the same sky draws, the quadrupole's combination of the generalized
    ORFs is the second moment of the per-source ORF over a uniform
    :math:`\cos\iota` and polarization angle; the inclination scatter
    :math:`c_\iota / (2/5)^2` the moment carries is the intensity second
    moment's, so it is divided out.
    """
    key, num_draws = jax.random.key(7), 40
    right_ascension, declination, _, _ = _isotropic_angles(key, num_draws)
    nodes, weights = gauss_legendre_nodes_weights(-1.0, 1.0, 8)
    polarization = (
        jnp.arange(NUM_POLARIZATION_NODES)[:, None]
        * jnp.pi
        / NUM_POLARIZATION_NODES
        * jnp.ones(num_draws)
    )
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
    flat = jnp.reshape(responses, (len(nodes), -1, 3 * RESPONSE_FREQUENCIES.size))
    moment = jnp.einsum("n,nli,nlj->ij", jnp.asarray(weights) / 2.0, flat, flat)
    orfs = generalized_overlap_reduction_function(
        key, num_draws, RESPONSE_FREQUENCIES, triangle
    )
    combined = sum(QUADRUPOLE_STOKES_WEIGHTS[pair] * orfs[pair] for pair in orfs)

    np.testing.assert_allclose(
        NORMALIZED_ORF_SCALE**2 * np.reshape(np.asarray(combined), moment.shape),
        RESPONSE_TO_CONNECTED * moment / (num_draws * NUM_POLARIZATION_NODES),
        rtol=1e-10,
        atol=1e-14,
    )


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
# The amplitude statistic
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def split_network(pair_leaves: dict[str, Any], response_draws: jax.Array) -> _Network:
    """Four generalized ORFs, each the moment of a disjoint quarter of the draws."""
    flat = np.reshape(np.asarray(response_draws), (NUM_DRAWS, -1))
    pairs: tuple[StokesPair, ...] = ("II", "QQ", "UU", "VV")
    orfs: dict[StokesPair, np.ndarray] = {
        pair: RESPONSE_TO_CONNECTED
        * (part.T @ part / NUM_DRAWS).reshape((NUM_PAIRS, NUM_BINS) * 2)
        for pair, part in zip(pairs, np.array_split(flat, 4), strict=True)
    }
    return _Network.from_leaves(**pair_leaves, generalized_orfs=orfs)


@pytest.fixture(scope="module")
def split_likelihood(
    model: CatalogModel, split_network: _Network
) -> DenseGaussianLikelihood[CatalogModel]:
    return DenseGaussianLikelihood._from_network(model, split_network, None)


def test_amplitude_weights_have_unit_response_to_the_template(
    likelihood: DenseGaussianLikelihood[CatalogModel],
) -> None:
    weights = amplitude_weights(likelihood, OFF_FIDUCIAL)

    assert float(weights @ likelihood.mean(OFF_FIDUCIAL)) == pytest.approx(
        1.0, rel=1e-12
    )


def test_amplitude_of_a_rescaled_template_is_the_scale(
    model: CatalogModel, network: _Network
) -> None:
    template = DenseGaussianLikelihood._from_network(model, network, None)
    scaled = DenseGaussianLikelihood._from_network(
        model, network, jnp.array([0.5, 1.3])[:, None] * template.mean(OFF_FIDUCIAL)
    )

    np.testing.assert_allclose(amplitude(scaled, OFF_FIDUCIAL), [0.5, 1.3], rtol=1e-12)


def test_amplitude_without_data_raises(model: CatalogModel, network: _Network) -> None:
    likelihood = DenseGaussianLikelihood._from_network(model, network, None)

    with pytest.raises(ValueError, match="without observed data"):
        amplitude(likelihood, OFF_FIDUCIAL)


def test_amplitude_variance_without_shot_noise_is_the_inverse_squared_snr(
    likelihood: DenseGaussianLikelihood[CatalogModel], network: _Network
) -> None:
    diagonal = DenseGaussianLikelihood._from_network(
        likelihood.model, network, None, shot_noise=False
    )
    squared_snr = jnp.sum((likelihood.mean(OFF_FIDUCIAL) / network.scale) ** 2)

    assert float(amplitude_variance(diagonal, OFF_FIDUCIAL)) == pytest.approx(
        float(1.0 / squared_snr), rel=1e-12
    )


def test_amplitude_variance_grows_with_shot_noise(
    likelihood: DenseGaussianLikelihood[CatalogModel], network: _Network
) -> None:
    diagonal = DenseGaussianLikelihood._from_network(
        likelihood.model, network, None, shot_noise=False
    )

    assert float(amplitude_variance(likelihood, OFF_FIDUCIAL)) > float(
        amplitude_variance(diagonal, OFF_FIDUCIAL)
    )


def test_amplitude_variance_budget_sums_to_the_amplitude_variance(
    split_likelihood: DenseGaussianLikelihood[CatalogModel],
) -> None:
    budget = amplitude_variance_budget(split_likelihood, OFF_FIDUCIAL)

    assert set(budget) == {"detector", "count", "intensity", "linear", "circular"}
    assert float(sum(budget.values())) == pytest.approx(
        float(amplitude_variance(split_likelihood, OFF_FIDUCIAL)), rel=1e-10
    )


def test_amplitude_variance_budget_without_generalized_orfs_raises(
    likelihood: DenseGaussianLikelihood[CatalogModel],
) -> None:
    with pytest.raises(ValueError, match="keep_generalized_orfs"):
        amplitude_variance_budget(likelihood, OFF_FIDUCIAL)


def test_covariance_count_term_is_the_template_over_the_expected_count(
    split_likelihood: DenseGaussianLikelihood[CatalogModel],
) -> None:
    r"""The count term is rank one along the mean, scaled by :math:`1/(\mathcal R T)`."""
    count = split_likelihood.covariance_components(OFF_FIDUCIAL)["count"]
    mean = split_likelihood.mean(OFF_FIDUCIAL)
    rate = split_likelihood.model(OFF_FIDUCIAL, shot_noise=False).merger_rate
    assert rate is not None
    expected_count = float(rate) * OBSERVATION_TIME * SECONDS_PER_YEAR

    assert float(mean @ count @ mean / (mean @ mean) ** 2) == pytest.approx(
        1.0 / expected_count, rel=1e-10
    )
    assert int(jnp.linalg.matrix_rank(count)) == 1
