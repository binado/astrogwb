"""The rescaled reference-redshift spectrum, its catalog, and its quadrature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb_mock_population import (
    EXTRINSIC_SITES,
    FIDUCIALS,
    INTRINSIC_DENSITY_SITES,
    bns_reference_catalog,
    build_reference_catalog,
    build_reference_spectrum,
    draw_bns_sources,
    mock_merger_rate,
    mock_population,
    population_models,
    source_draw_model,
    uniform_bns_log_prob,
    uniform_source_model,
)
from numpyro import handlers
from tolerances import assert_close

from astrogwb.background.importance.catalog import ReferenceCatalog, intensity_data
from astrogwb.background.importance.model import CatalogModel
from astrogwb.background.importance.spectrum import (
    INCLINATION_MEAN,
    INCLINATION_RAW_SECOND_MOMENT,
    REFERENCE_INCLINATION,
    phinney_kernel,
    redshift_quadrature,
    reference_catalog,
)
from astrogwb.background.simulation import realization_key
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import log_gw_em_ratio, luminosity_distance
from astrogwb.distributions.mass import MaxOfTwoUniformsDistribution
from astrogwb.rates import differential_merger_rate, madau_dickinson_rate
from astrogwb.utils import gauss_legendre_nodes_weights
from astrogwb.utils.numpyro import Sampling, sample_and_logprob
from astrogwb.waveform import WaveformMetadata, inclination_factor, waveform_generator

#: Relative second moment <P^2>/<P>^2 for unit-mean inclination response draws.
INCLINATION_SECOND_MOMENT = 355.0 / 252.0

#: The observed grid of the comparisons: four 10 Hz bins from 20 Hz.
OBSERVED_WAVEFORM = WaveformMetadata(
    approximant="AnalyticInspiral",
    minimum_frequency=20.0,
    maximum_frequency=50.0,
    reference_frequency=20.0,
    sampling_frequency=128.0,
    frequency_resolution=10.0,
)

#: Redshift nodes of the comparisons.
NUM_NODES = 3

#: Observation time of the shot-noise comparisons, in years.
OBSERVATION_TIME = 1.0


def _covariance(model: CatalogModel, params: dict[str, Any]) -> np.ndarray:
    """The physical covariance of a realization over :data:`OBSERVATION_TIME`, ``(F, F)``."""
    seconds = OBSERVATION_TIME * SECONDS_PER_YEAR
    return np.asarray(model.intensity_second_moment(params)) / seconds


def _bin_variance(model: CatalogModel, params: dict[str, Any]) -> np.ndarray:
    """The per-bin shot-noise variance, ``(F,)``."""
    return np.diag(_covariance(model, params))


@dataclass(frozen=True)
class ReferenceSetup:
    """What a test needs to draw and bind one reference catalog."""

    waveform: WaveformMetadata
    fiducials: dict[str, float]
    num_samples: int
    minimum_redshift: float
    maximum_redshift: float
    n_grid: int


@pytest.fixture
def reference_setup() -> ReferenceSetup:
    """Six closed-form draws, on a fine log grid covering every rescaled query.

    Queries reach ``50 Hz * (1 + 5) / (1 + 0.1)``, about 273 Hz; the analytic
    inspiral ends far above that for these masses, so a comparison sees only
    the rescaling and the interpolation.
    """
    return ReferenceSetup(
        waveform=OBSERVED_WAVEFORM.model_copy(
            update={
                "frequency_spacing": "log",
                "maximum_frequency": 300.0,
                "frequency_resolution": 0.02,
            }
        ),
        fiducials={
            "H0": 67.66,
            "Omega_m": 0.3096,
            "gamma": 1.42,
            "kappa": 4.62,
            "z_peak": 1.84,
            "local_merger_rate": 770.0,
            "minimum_mass": 1.0,
            "mass_width": 1.5,
        },
        num_samples=6,
        minimum_redshift=0.1,
        maximum_redshift=5.0,
        # Fine: the brute force reads the table, the spectrum the closed
        # form, and they agree only to the table's interpolation error.
        n_grid=8192,
    )


def _samples(
    setup: ReferenceSetup,
    key: jax.Array,
    *,
    sampling: Sampling = "random",
    num_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Intrinsic draws of the mock population at the setup's fiducials."""
    return draw_bns_sources(
        setup.fiducials,
        key,
        setup.num_samples if num_samples is None else num_samples,
        sampling=sampling,
    )


def _catalog(
    setup: ReferenceSetup,
    key: jax.Array,
    *,
    samples: dict[str, np.ndarray] | None = None,
    reference_redshift: float | None = None,
    waveform: WaveformMetadata | None = None,
) -> ReferenceCatalog:
    """The reference catalog of ``samples``, at the window's lower edge by default."""
    return bns_reference_catalog(
        waveform_generator(setup.waveform if waveform is None else waveform),
        _samples(setup, key) if samples is None else samples,
        setup.fiducials,
        reference_redshift=(
            setup.minimum_redshift if reference_redshift is None else reference_redshift
        ),
    )


def _spectrum(
    catalog: ReferenceCatalog,
    setup: ReferenceSetup,
    *,
    minimum_redshift: float | None = None,
    **settings: Any,
) -> CatalogModel:
    """``catalog`` bound to the mock target on the setup's window and observed grid."""
    settings.setdefault(
        "frequencies", np.asarray(waveform_generator(OBSERVED_WAVEFORM).frequencies)
    )
    settings.setdefault("num_redshift_nodes", NUM_NODES)
    return build_reference_spectrum(
        catalog,
        merger_rate=mock_merger_rate,
        minimum_redshift=(
            setup.minimum_redshift if minimum_redshift is None else minimum_redshift
        ),
        maximum_redshift=setup.maximum_redshift,
        **settings,
    )


def test_reference_inclination_is_face_on() -> None:
    assert float(inclination_factor(REFERENCE_INCLINATION)) == pytest.approx(
        2.0, rel=1e-14
    )


@pytest.mark.integration
def test_importance_path_does_not_construct_a_redshift_distribution(
    reference_setup: ReferenceSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("importance should not construct a redshift distribution")

    monkeypatch.setattr(
        "astrogwb.distributions.redshift.base.RedshiftDistribution.__init__", fail
    )
    catalog = _catalog(reference_setup, realization_key(41, 0))
    model = _spectrum(catalog, reference_setup, reweight_masses=False)
    assert np.all(np.isfinite(model.spectral_density(reference_setup.fiducials)))


def test_inclination_second_moment_is_the_isotropic_ratio_of_moments() -> None:
    # g is a polynomial of degree four in cos(iota), so five nodes integrate
    # g**2 exactly under the isotropic (uniform in cos) average.
    cosine, weights = gauss_legendre_nodes_weights(-1.0, 1.0, 5)
    factor = np.asarray(inclination_factor(np.arccos(np.asarray(cosine))))
    second_moment = np.sum(np.asarray(weights) * factor**2) / 2.0
    assert second_moment / 0.8**2 == pytest.approx(INCLINATION_SECOND_MOMENT, rel=1e-13)


@pytest.mark.parametrize("power", [2, 3, 5])
def test_redshift_quadrature_polynomial_in_scale_factor_is_exact(power: int) -> None:
    # dz = da / a**2, so a**power integrates as a polynomial of degree
    # power - 2 in a, which four nodes reproduce up to degree seven.
    minimum, maximum = 0.01, 5.0
    redshift, weights = redshift_quadrature(minimum, maximum, 4)
    lower, upper = 1.0 / (1.0 + maximum), 1.0 / (1.0 + minimum)
    exact = (upper ** (power - 1) - lower ** (power - 1)) / (power - 1)
    estimate = np.sum(weights / (1.0 + redshift) ** power)
    assert estimate == pytest.approx(exact, rel=1e-13)


@pytest.mark.integration
def test_sobol_reference_catalog_stratifies_the_primary_mass(
    reference_setup: ReferenceSetup,
) -> None:
    """Pinned sites take no coordinate, so the primary mass takes the first."""
    key = realization_key(41, 0)
    samples = _samples(reference_setup, key, sampling="sobol", num_samples=8)
    catalog = _catalog(reference_setup, key, samples=samples)

    assert not set(EXTRINSIC_SITES) & catalog.intrinsic.keys()
    assert catalog.reference_redshift == 0.1
    np.testing.assert_array_equal(samples["inclination"], REFERENCE_INCLINATION)
    fiducials = reference_setup.fiducials
    primary = MaxOfTwoUniformsDistribution(
        fiducials["minimum_mass"], fiducials["mass_width"]
    )
    cells = np.floor(
        np.asarray(primary.cdf(np.asarray(catalog.intrinsic["source_frame_mass_1"])))
        * 8
    )
    np.testing.assert_array_equal(np.sort(cells), np.arange(8))


@pytest.mark.parametrize("site", ["redshift", "luminosity_distance", "inclination"])
def test_reference_catalog_rejects_reserved_intrinsic_sites(
    reference_setup: ReferenceSetup, site: str
) -> None:
    samples = _samples(reference_setup, realization_key(41, 0))
    with pytest.raises(ValueError, match="intrinsic_sites must exclude"):
        reference_catalog(
            waveform_generator(reference_setup.waveform),
            samples,
            intrinsic_sites=(*INTRINSIC_DENSITY_SITES, site),
            proposal_log_prob=np.zeros(reference_setup.num_samples),
            reference_redshift=0.1,
            parameters=reference_setup.fiducials,
        )


def test_reference_catalog_rejects_a_misshapen_proposal_log_prob(
    reference_setup: ReferenceSetup,
) -> None:
    samples = _samples(reference_setup, realization_key(41, 0))
    with pytest.raises(ValueError, match="proposal_log_prob must have shape"):
        reference_catalog(
            waveform_generator(reference_setup.waveform),
            samples,
            intrinsic_sites=INTRINSIC_DENSITY_SITES,
            proposal_log_prob=np.zeros(reference_setup.num_samples + 1),
            reference_redshift=0.1,
            parameters=reference_setup.fiducials,
        )


def _spectrum_on_nodes(
    setup: ReferenceSetup,
    key: jax.Array,
    params: dict[str, float],
    density_sites: tuple[str, ...],
    *,
    covariance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """The quadrature by brute force: every draw generated at every node.

    No rescaling: the waveform is generated in the observer frame at each node,
    at the fiducial distance, and the redshift kernel is read off the target's
    own model row by row. Returns the spectrum and its shot-noise variance over
    :data:`OBSERVATION_TIME`, the latter from each node's power at the target
    distance, squared.
    """
    population = mock_population()
    minimum, maximum = setup.minimum_redshift, setup.maximum_redshift
    redshift, weights = redshift_quadrature(minimum, maximum, NUM_NODES)
    fiducial_model = source_draw_model(
        *population_models(
            population,
            setup.fiducials,
            minimum_redshift=minimum,
            maximum_redshift=maximum,
            n_grid=setup.n_grid,
        )
    )
    samples = sample_and_logprob(
        handlers.condition(
            handlers.condition(
                fiducial_model,
                data={"redshift": minimum, "inclination": REFERENCE_INCLINATION},
            ),
            data={name: 0.0 for name in EXTRINSIC_SITES},
        ),
        key,
        num_samples=setup.num_samples,
        density_sites=(),
    )[0]
    # Every source at every node, node-major: row ``j * N + k`` is source ``k``
    # at node ``j``, replayed through the model so redshift-derived columns
    # (the luminosity distance) are recomputed.
    count = setup.num_samples
    tiled = {
        name: np.tile(np.asarray(values), redshift.size)
        for name, values in samples.items()
    }
    tiled["redshift"] = np.repeat(redshift, count)
    tiled["inclination"] = np.full(count * redshift.size, REFERENCE_INCLINATION)
    outputs, _ = sample_and_logprob(fiducial_model, density_sites=(), values=tiled)
    rows = {name: np.asarray(values) for name, values in outputs.items()}
    # Independent inclination integral: generate actual waveforms at each
    # Gauss-Legendre node, rather than using the production moment constants.
    cosine, orientation_weights = np.polynomial.legendre.leggauss(5)
    generator = waveform_generator(OBSERVED_WAVEFORM)
    powers = np.stack(
        [
            intensity_data(
                generator,
                {**rows, "inclination": np.full(count * redshift.size, np.arccos(c))},
            )["intensity"]
            for c in cosine
        ]
    ).reshape(5, -1, NUM_NODES, count)
    power = np.einsum("l,lfzn->fzn", orientation_weights / 2, powers)
    squared_power = np.einsum("l,lfzn->fzn", orientation_weights / 2, powers**2)

    redshift_distribution, source_model = population_models(
        population,
        params,
        minimum_redshift=minimum,
        maximum_redshift=maximum,
        n_grid=setup.n_grid,
    )
    merger_rate = redshift_distribution.total_merger_rate()
    model = source_draw_model(redshift_distribution, source_model)
    outputs, log_density = sample_and_logprob(
        model, density_sites=("redshift",), values=rows
    )
    distance_ratio = rows["luminosity_distance"] / outputs["luminosity_distance"]
    target_power = power * distance_ratio.reshape(NUM_NODES, 6) ** 2
    density = weights * np.exp(log_density[::6])
    first = {name: values[:6] for name, values in rows.items()}
    _, target = sample_and_logprob(model, density_sites=density_sites, values=first)
    _, drawn = sample_and_logprob(
        fiducial_model, density_sites=density_sites, values=first
    )
    importance = np.exp(np.asarray(target) - np.asarray(drawn))
    rate = float(merger_rate)
    spectrum = rate * np.einsum("fzn,z,n->f", target_power, density, importance) / 6
    target_squared_power = squared_power * distance_ratio.reshape(NUM_NODES, 6) ** 4
    second_moment = (
        np.einsum("fzn,z,n->f", target_squared_power, density, importance) / 6
    )
    variance = rate * second_moment / (OBSERVATION_TIME * SECONDS_PER_YEAR)
    if covariance:
        target_powers = powers * distance_ratio.reshape(NUM_NODES, count) ** 2
        raw_product = (
            np.einsum(
                "l,lfzn,lgzn,z,n->fg",
                orientation_weights / 2,
                target_powers,
                target_powers,
                density,
                importance,
            )
            / count
        )
        return spectrum, rate * raw_product / (OBSERVATION_TIME * SECONDS_PER_YEAR)
    return spectrum, variance


def _rescaled_and_brute_force(
    setup: ReferenceSetup,
    density_sites: tuple[str, ...],
    points: list[dict[str, float]],
    *,
    variance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescaled and brute-force spectra, or variances, on the same draws and nodes."""
    key = realization_key(41, 0)
    model = _spectrum(_catalog(setup, key), setup, reweight_masses=bool(density_sites))
    rescaled = [
        np.asarray(_bin_variance(model, p) if variance else model.spectral_density(p))
        for p in points
    ]
    brute_force = [
        _spectrum_on_nodes(setup, key, p, density_sites)[int(variance)] for p in points
    ]
    return np.stack(rescaled), np.stack(brute_force)


@pytest.mark.integration
@pytest.mark.parametrize("variance", [False, True], ids=["spectrum", "shot-noise"])
def test_rescaled_prediction_matches_waveforms_generated_at_every_node(
    reference_setup: ReferenceSetup, variance: bool
) -> None:
    fiducials = reference_setup.fiducials
    points = [
        fiducials,
        {**fiducials, "H0": 60.0},
        {**fiducials, "xi_0": 1.2, "xi_n": 2.0},
    ]
    rescaled, brute_force = _rescaled_and_brute_force(
        reference_setup, (), points, variance=variance
    )

    # Linear interpolation of an f^(-7/3) power law on a 0.1% log grid.
    assert_close(rescaled, brute_force)


@pytest.mark.integration
@pytest.mark.parametrize("variance", [False, True], ids=["spectrum", "shot-noise"])
def test_rescaled_prediction_reweights_like_waveforms_generated_at_every_node(
    reference_setup: ReferenceSetup, variance: bool
) -> None:
    sites = ("source_frame_mass_1", "source_frame_mass_2")
    points = [{**reference_setup.fiducials, "minimum_mass": 0.9, "mass_width": 1.6}]
    rescaled, brute_force = _rescaled_and_brute_force(
        reference_setup, sites, points, variance=variance
    )

    assert_close(rescaled, brute_force)


@pytest.mark.integration
def test_explicit_bns_weights_at_proposal_and_shifted_support(
    reference_setup: ReferenceSetup,
) -> None:
    fiducials = reference_setup.fiducials
    key = realization_key(41, 0)
    samples = _samples(reference_setup, key, sampling="sobol", num_samples=32)
    reference = _catalog(reference_setup, key, samples=samples)
    source_model = uniform_source_model

    frequencies = np.asarray(waveform_generator(OBSERVED_WAVEFORM).frequencies)
    settings: dict[str, Any] = {
        "frequencies": frequencies,
        "num_redshift_nodes": NUM_NODES,
        "minimum_redshift": reference_setup.minimum_redshift,
        "maximum_redshift": reference_setup.maximum_redshift,
        "merger_rate": mock_merger_rate,
    }
    explicit = CatalogModel.from_catalog(
        reference, log_prob=uniform_bns_log_prob, **settings
    )
    cached = CatalogModel.from_catalog(
        reference, target_matches_proposal=True, **settings
    )
    predict = jax.jit(CatalogModel.__call__)
    weights = jax.jit(CatalogModel.log_weights)
    cumulants = jax.jit(
        lambda model, params, projectors: model.projected_cumulants(
            params, projectors, observation_time=OBSERVATION_TIME
        )
    )
    projectors = jnp.ones((1, frequencies.size))
    narrow = {**fiducials, "mass_width": 1.0}
    assert cached.node_moments is not None
    np.testing.assert_array_equal(weights(cached, fiducials), 0)

    def traced_density(params: dict[str, float]) -> jax.Array:
        _, log_prob = sample_and_logprob(
            source_model(params),
            density_sites=INTRINSIC_DENSITY_SITES,
            values=samples,
        )
        return log_prob

    shifted_weights = weights(explicit, narrow)
    assert np.any(np.isneginf(shifted_weights))
    assert np.any(np.isfinite(shifted_weights))
    with pytest.warns(UserWarning, match="Out-of-support"):
        expected_weights = traced_density(narrow) - traced_density(fiducials)
    assert_close(shifted_weights, expected_weights)

    for result, baseline in zip(
        jax.tree.leaves(predict(explicit, fiducials)),
        jax.tree.leaves(predict(cached, fiducials)),
        strict=True,
    ):
        assert_close(result, baseline)
    assert_close(
        cumulants(explicit, fiducials, projectors),
        cumulants(cached, fiducials, projectors),
    )
    np.testing.assert_array_equal(
        predict(explicit, narrow).extras["total_merger_rate"],
        predict(explicit, fiducials).extras["total_merger_rate"],
    )


@pytest.fixture
def shot_noise_model(reference_setup: ReferenceSetup) -> CatalogModel:
    """One reference catalog bound to its own proposal."""
    catalog = _catalog(reference_setup, realization_key(41, 0))
    return _spectrum(catalog, reference_setup, reweight_masses=False)


@pytest.mark.integration
def test_rescaled_shot_noise_grows_linearly_with_the_hubble_constant(
    reference_setup: ReferenceSetup,
    shot_noise_model: CatalogModel,
) -> None:
    """The squared kernel goes as :math:`D_H / \\chi^2 \\propto H_0`."""
    hubble_constant = 90.0
    fiducials = reference_setup.fiducials
    model = shot_noise_model
    shifted = np.asarray(_bin_variance(model, {**fiducials, "H0": hubble_constant}))
    ratio = shifted / np.asarray(_bin_variance(model, fiducials))

    assert_close(ratio, np.full(ratio.shape, hubble_constant / fiducials["H0"]))


@pytest.mark.parametrize("xi_0", [1.0, 0.8])
def test_phinney_kernel_times_the_distance_ratio_squared_is_the_rate_density(
    xi_0: float,
) -> None:
    """Undoing the distance factor leaves ``psi / (1 + z) dV_c/dz`` per second."""
    redshift, _ = redshift_quadrature(0.1, 5.0, 24)
    hubble_constant, omega_m, reference_distance = 67.66, 0.3096, 700.0
    merger_rate = np.asarray(madau_dickinson_rate(redshift, 1.42, 4.62, 1.84, 770.0))
    distance_ratio = xi_0 + (1.0 - xi_0) * (1.0 + redshift) ** -2.0

    kernel = phinney_kernel(
        redshift,
        merger_rate,
        distance_ratio,
        hubble_constant,
        omega_m,
        reference_distance,
    )
    distance = distance_ratio * luminosity_distance(redshift, hubble_constant, omega_m)
    undone = np.asarray(kernel) * (np.asarray(distance) / reference_distance) ** 2

    assert_close(
        undone,
        differential_merger_rate(redshift, merger_rate, hubble_constant, omega_m),
    )


@pytest.mark.parametrize(
    ("parameter", "exponent"), [("H0", -1.0), ("local_merger_rate", 1.0)]
)
def test_spectrum_scales_as_a_power_of_h0_and_the_local_merger_rate(
    parameter: str, exponent: float
) -> None:
    """``S ~ 1/H0`` and ``S ~ R0``: the exponents the cosmology and kernel imply."""
    spectrum = build_reference_spectrum(
        build_reference_catalog(num_sources=16), merger_rate=mock_merger_rate
    ).spectral_density
    fiducial = FIDUCIALS[parameter]
    factor = 1.3

    baseline = spectrum(FIDUCIALS)
    scaled = spectrum({**FIDUCIALS, parameter: factor * fiducial})

    assert_close(scaled, factor**exponent * baseline)


def _bound_predictions(
    catalog: ReferenceCatalog,
    setup: ReferenceSetup,
    *,
    minimum_redshift: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The spectrum and shot-noise variance of ``catalog`` at the fiducials."""
    model = _spectrum(
        catalog, setup, minimum_redshift=minimum_redshift, reweight_masses=False
    )
    return (
        np.asarray(model.spectral_density(setup.fiducials)),
        np.asarray(_bin_variance(model, setup.fiducials)),
    )


@pytest.mark.integration
@pytest.mark.parametrize("window_edge", [0.5, 1.5])
def test_low_reference_catalog_matches_one_generated_at_the_window_edge(
    reference_setup: ReferenceSetup, window_edge: float
) -> None:
    """Same key, hence same intrinsic draws: only the interpolation differs."""
    key = realization_key(41, 0)
    low = _catalog(reference_setup, key)
    direct = _catalog(reference_setup, key, reference_redshift=window_edge)
    for name in INTRINSIC_DENSITY_SITES:
        np.testing.assert_array_equal(low.intrinsic[name], direct.intrinsic[name])

    rescaled = _bound_predictions(low, reference_setup, minimum_redshift=window_edge)
    expected = _bound_predictions(direct, reference_setup, minimum_redshift=window_edge)

    # Linear interpolation of an f^(-7/3) power law on a 0.1% log grid, as in
    # the comparisons against waveforms generated at every node.
    assert_close(rescaled[0], expected[0])
    assert_close(rescaled[1], expected[1])


@pytest.mark.integration
@pytest.mark.parametrize("reference", [0.02, 0.05, 0.1])
def test_reference_catalog_distance_is_at_the_reference_redshift(
    reference_setup: ReferenceSetup, reference: float
) -> None:
    """The distance is the cosmological one at the reference, not at the window edge."""
    catalog = _catalog(
        reference_setup, realization_key(41, 0), reference_redshift=reference
    )
    fiducials = reference_setup.fiducials

    expected = luminosity_distance(reference, fiducials["H0"], fiducials["Omega_m"])
    assert catalog.reference_redshift == reference
    assert catalog.reference_distance == pytest.approx(float(expected), rel=1e-12)


@pytest.mark.integration
def test_reference_catalog_distance_carries_the_modified_propagation_ratio(
    reference_setup: ReferenceSetup,
) -> None:
    redshift, xi_0, xi_n = 0.1, 1.2, 2.0
    modified = {**reference_setup.fiducials, "xi_0": xi_0, "xi_n": xi_n}
    catalog = bns_reference_catalog(
        waveform_generator(reference_setup.waveform),
        _samples(reference_setup, realization_key(41, 0)),
        modified,
        reference_redshift=redshift,
    )
    plain = luminosity_distance(redshift, modified["H0"], modified["Omega_m"])
    expected = float(plain * np.exp(log_gw_em_ratio(redshift, xi_0, xi_n)))
    assert catalog.reference_distance == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("moment", [1, 2])
def test_face_on_moments_match_independent_inclination_integral(moment: int) -> None:
    cosine, weights = np.polynomial.legendre.leggauss(8)
    relative_power = np.asarray(inclination_factor(np.arccos(cosine))) / float(
        inclination_factor(0.0)
    )
    expected = (INCLINATION_MEAN, INCLINATION_RAW_SECOND_MOMENT)[moment - 1]
    assert np.dot(weights / 2, relative_power**moment) == pytest.approx(expected)


@pytest.mark.integration
@pytest.mark.parametrize("moment", [1, 2])
@pytest.mark.parametrize(
    "approximant", ["AnalyticInspiral", "TaylorF2", "IMRPhenomXAS_NRTidalv3"]
)
def test_face_on_and_mean_inclination_waveforms_give_equivalent_moments(
    reference_setup: ReferenceSetup, moment: int, approximant: str
) -> None:
    waveform = reference_setup.waveform.model_copy(update={"approximant": approximant})
    samples = _samples(reference_setup, realization_key(41, 0))
    catalog = _catalog(reference_setup, realization_key(41, 0), waveform=waveform)
    effective_inclination = float(np.arccos(np.sqrt(np.sqrt(11.2) - 3.0)))
    columns = {
        **samples,
        "redshift": np.full(reference_setup.num_samples, catalog.reference_redshift),
        "luminosity_distance": np.full(
            reference_setup.num_samples, catalog.reference_distance
        ),
        "inclination": np.full(reference_setup.num_samples, effective_inclination),
    }
    effective = intensity_data(waveform_generator(waveform), columns)
    old_factor = 1.0 if moment == 1 else INCLINATION_SECOND_MOMENT
    new_factor = INCLINATION_MEAN if moment == 1 else INCLINATION_RAW_SECOND_MOMENT
    assert_close(
        new_factor * catalog.intensity**moment,
        old_factor * effective["intensity"] ** moment,
    )


@pytest.mark.parametrize("approximant", ["IMRPhenomHM", "IMRPhenomXP"])
def test_reference_conversion_rejects_higher_modes_and_precession(
    reference_setup: ReferenceSetup, approximant: str
) -> None:
    waveform = reference_setup.waveform.model_copy(update={"approximant": approximant})
    with pytest.raises(ValueError, match="dominant-quadrupole"):
        _catalog(reference_setup, realization_key(41, 0), waveform=waveform)


@pytest.mark.integration
@pytest.mark.parametrize(
    "density_sites", [(), ("source_frame_mass_1", "source_frame_mass_2")]
)
def test_dense_covariance_matches_physical_inclination_quadrature(
    reference_setup: ReferenceSetup, density_sites: tuple[str, ...]
) -> None:
    key = realization_key(41, 0)
    params = {**reference_setup.fiducials, "minimum_mass": 0.9, "mass_width": 1.6}
    model = _spectrum(
        _catalog(reference_setup, key),
        reference_setup,
        reweight_masses=bool(density_sites),
    )
    _, expected = _spectrum_on_nodes(
        reference_setup, key, params, density_sites, covariance=True
    )
    # The reference power uses log-frequency interpolation; the oracle generates
    # each waveform at its actual redshift. This bound covers that grid error.
    assert_close(_covariance(model, params), expected)


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [(0.05, 1.0, "below the reference redshift"), (0.5, 0.5, "redshift bounds")],
)
def test_catalog_model_rejects_invalid_integration_bounds(
    reference_setup: ReferenceSetup,
    minimum: float,
    maximum: float,
    message: str,
) -> None:
    catalog = ReferenceCatalog(
        frequencies=np.array([20.0, 30.0]),
        intensity=np.ones((2, 6)),
        intrinsic={},
        proposal_log_prob=np.zeros(6),
        reference_redshift=0.1,
        reference_distance=475.0,
    )
    with pytest.raises(ValueError, match=message):
        CatalogModel.from_catalog(
            catalog,
            merger_rate=mock_merger_rate,
            frequencies=np.array([20.0]),
            num_redshift_nodes=3,
            minimum_redshift=minimum,
            maximum_redshift=maximum,
            target_matches_proposal=True,
        )
