"""The rescaled reference-redshift spectrum, its catalog, and its quadrature."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    build_reference_catalog,
    build_reference_spectrum,
)

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import luminosity_distance
from astrogwb.distributions.mass import MaxOfTwoUniformsDistribution
from astrogwb.distributions.rates import madau_dickinson_rate, total_merger_rate
from astrogwb.gwb.importance import (
    EFFECTIVE_INCLINATION,
    INCLINATION_SECOND_MOMENT,
    _pin_redshift_and_inclination,
    build_rescaled_shot_noise,
    build_rescaled_spectrum,
    phinney_kernel,
    redshift_quadrature,
    reference_catalog,
)
from astrogwb.populations import PopulationMetadata, build_population, joint_model
from astrogwb.populations.evaluation import evaluate_sources, sample_sources
from astrogwb.simulators.core import batch_keys
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    polarization_power_data,
)
from astrogwb.utils import gauss_legendre_nodes_weights
from astrogwb.waveform import WaveformMetadata
from astrogwb.waveform.generator.analytical import inclination_factor

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


@pytest.fixture
def reference_metadata() -> CatalogMetadata:
    """Six closed-form draws, on a fine log grid covering every rescaled query.

    Queries reach ``50 Hz * (1 + 5) / (1 + 0.1)``, about 273 Hz; the analytic
    inspiral ends far above that for these masses, so a comparison sees only
    the rescaling and the interpolation.
    """
    return CatalogMetadata(
        waveform=OBSERVED_WAVEFORM.model_copy(
            update={
                "frequency_spacing": "log",
                "maximum_frequency": 300.0,
                "frequency_resolution": 0.02,
            }
        ),
        population=PopulationMetadata(
            model_name="bns_coba",
            model_kwargs={
                "mass_model": "uniform",
                "minimum_redshift": 0.1,
                "maximum_redshift": 5.0,
                # Fine: the brute force reads the table, the spectrum the closed
                # form, and they agree only to the table's interpolation error.
                "n_grid": 8192,
            },
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
    )


def test_effective_inclination_gives_the_isotropic_mean_factor() -> None:
    assert float(inclination_factor(EFFECTIVE_INCLINATION)) == pytest.approx(
        0.8, rel=1e-14
    )


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


def test_redshift_quadrature_weights_sum_to_the_window_width() -> None:
    _, weights = redshift_quadrature(0.01, 5.0, 32)
    assert np.sum(weights) == pytest.approx(4.99, rel=1e-10)


def test_redshift_quadrature_nodes_increase_inside_the_window() -> None:
    redshift, weights = redshift_quadrature(0.01, 5.0, 16)
    assert np.all(np.diff(redshift) > 0.0)
    assert 0.01 < redshift[0] and redshift[-1] < 5.0
    assert np.all(weights > 0.0)


def test_redshift_quadrature_without_nodes_raises() -> None:
    with pytest.raises(ValueError, match="num_nodes"):
        redshift_quadrature(0.01, 5.0, 0)


@pytest.mark.integration
def test_reference_catalog_places_every_draw_at_the_minimum_redshift(
    reference_metadata: CatalogMetadata,
) -> None:
    data = reference_catalog(reference_metadata, batch_keys(41, 1)[0])

    columns = data["source_parameters"]
    assert data["polarization_power"].shape == (data["frequencies"].size, 6)
    np.testing.assert_array_equal(columns["redshift"], np.full(6, 0.1))
    np.testing.assert_array_equal(
        columns["inclination"], np.full(6, EFFECTIVE_INCLINATION)
    )


@pytest.mark.integration
def test_sobol_reference_catalog_stratifies_the_primary_mass(
    reference_metadata: CatalogMetadata,
) -> None:
    """Pinned sites take no coordinate, so the primary mass takes the first."""
    metadata = CatalogMetadata.model_validate(
        {**reference_metadata.model_dump(), "num_samples": 8, "sampling": "sobol"}
    )
    data = reference_catalog(metadata, batch_keys(41, 1)[0])

    columns = data["source_parameters"]
    np.testing.assert_array_equal(columns["redshift"], np.full(8, 0.1))
    np.testing.assert_array_equal(
        columns["inclination"], np.full(8, EFFECTIVE_INCLINATION)
    )
    fiducials = metadata.fiducials
    primary = MaxOfTwoUniformsDistribution(
        fiducials["minimum_mass"], fiducials["mass_width"]
    )
    cells = np.floor(
        np.asarray(primary.cdf(np.asarray(columns["source_frame_mass_1"]))) * 8
    )
    np.testing.assert_array_equal(np.sort(cells), np.arange(8))


def _spectrum_on_nodes(
    metadata: CatalogMetadata,
    key: jax.Array,
    params: dict[str, float],
    density_sites: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """The quadrature by brute force: every draw generated at every node.

    No rescaling: the waveform is generated in the observer frame at each node,
    at the fiducial distance, and the redshift kernel is read off the target's
    own model row by row. Returns the spectrum and its shot-noise variance over
    :data:`OBSERVATION_TIME`, the latter from each node's power at the target
    distance, squared.
    """
    population = metadata.population
    minimum, maximum = (
        float(population.model_kwargs[name])
        for name in ("minimum_redshift", "maximum_redshift")
    )
    redshift, weights = redshift_quadrature(minimum, maximum, NUM_NODES)
    fiducial_model = joint_model(*population.build()(metadata.fiducials))
    samples = sample_sources(
        _pin_redshift_and_inclination(fiducial_model, minimum),
        key,
        num_samples=metadata.num_samples,
    )
    # Every source at every node, node-major: row ``j * N + k`` is source ``k``
    # at node ``j``, replayed through the model so redshift-derived columns
    # (the luminosity distance) are recomputed.
    count = metadata.num_samples
    tiled = {
        name: np.tile(np.asarray(values), redshift.size)
        for name, values in samples.items()
    }
    tiled["redshift"] = np.repeat(redshift, count)
    tiled["inclination"] = np.full(count * redshift.size, EFFECTIVE_INCLINATION)
    _, outputs = evaluate_sources(fiducial_model, tiled, density_sites=())
    rows = {name: np.asarray(values) for name, values in outputs.items()}
    power = polarization_power_data(OBSERVED_WAVEFORM.build(), rows)
    power = np.asarray(power["polarization_power"]).reshape(-1, NUM_NODES, 6)

    redshift_distribution, source_model = population.build()(params)
    merger_rate = redshift_distribution.total_merger_rate()
    model = joint_model(redshift_distribution, source_model)
    log_density, outputs = evaluate_sources(model, rows, density_sites=("redshift",))
    distance_ratio = rows["luminosity_distance"] / outputs["luminosity_distance"]
    target_power = power * distance_ratio.reshape(NUM_NODES, 6) ** 2
    density = weights * np.exp(log_density[::6])
    first = {name: values[:6] for name, values in rows.items()}
    target, _ = evaluate_sources(model, first, density_sites=density_sites)
    drawn, _ = evaluate_sources(fiducial_model, first, density_sites=density_sites)
    importance = np.exp(np.asarray(target) - np.asarray(drawn))
    rate = float(merger_rate)
    spectrum = rate * np.einsum("fzn,z,n->f", target_power, density, importance) / 6
    second_moment = np.einsum("fzn,z,n->f", target_power**2, density, importance) / 6
    variance = (
        INCLINATION_SECOND_MOMENT
        * rate
        * second_moment
        / (OBSERVATION_TIME * SECONDS_PER_YEAR)
    )
    return spectrum, variance


def _rescaled_and_brute_force(
    metadata: CatalogMetadata,
    density_sites: tuple[str, ...],
    points: list[dict[str, float]],
    *,
    variance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescaled and brute-force spectra, or variances, on the same draws and nodes."""
    key = batch_keys(41, 1)[0]
    reference = reference_catalog(metadata, key)
    target = build_population(
        metadata.population.model_name, **metadata.population.model_kwargs
    )
    settings: dict[str, Any] = {
        "population": target,
        "frequencies": OBSERVED_WAVEFORM.build().frequencies,
        "num_redshift_nodes": NUM_NODES,
        "density_sites": density_sites,
    }
    if variance:
        variance_fn = build_rescaled_shot_noise(
            reference, metadata, observation_time=OBSERVATION_TIME, **settings
        )
        rescaled = [np.asarray(variance_fn(p)) for p in points]
    else:
        spectrum_fn, _ = build_rescaled_spectrum(reference, metadata, **settings)
        rescaled = [np.asarray(spectrum_fn(p)[0]) for p in points]
    brute_force = [
        _spectrum_on_nodes(metadata, key, p, density_sites)[int(variance)]
        for p in points
    ]
    return np.stack(rescaled), np.stack(brute_force)


@pytest.mark.integration
def test_rescaled_spectrum_matches_waveforms_generated_at_every_node(
    reference_metadata: CatalogMetadata,
) -> None:
    fiducials = reference_metadata.fiducials
    points = [
        fiducials,
        {**fiducials, "H0": 60.0},
        {**fiducials, "xi_0": 1.2, "xi_n": 2.0},
    ]
    rescaled, brute_force = _rescaled_and_brute_force(reference_metadata, (), points)

    # Linear interpolation of an f^(-7/3) power law on a 0.1% log grid.
    np.testing.assert_allclose(rescaled, brute_force, rtol=1e-5)


@pytest.mark.integration
def test_rescaled_spectrum_reweights_like_waveforms_generated_at_every_node(
    reference_metadata: CatalogMetadata,
) -> None:
    sites = ("source_frame_mass_1", "source_frame_mass_2")
    points = [{**reference_metadata.fiducials, "minimum_mass": 0.9, "mass_width": 1.6}]
    rescaled, brute_force = _rescaled_and_brute_force(reference_metadata, sites, points)

    np.testing.assert_allclose(rescaled, brute_force, rtol=1e-5)


@pytest.mark.integration
def test_rescaled_shot_noise_matches_waveforms_generated_at_every_node(
    reference_metadata: CatalogMetadata,
) -> None:
    fiducials = reference_metadata.fiducials
    points = [
        fiducials,
        {**fiducials, "H0": 60.0},
        {**fiducials, "xi_0": 1.2, "xi_n": 2.0},
    ]
    rescaled, brute_force = _rescaled_and_brute_force(
        reference_metadata, (), points, variance=True
    )

    # Squaring doubles the power-law index, so linear interpolation in ln f
    # errs several times more than for the spectrum: about 2e-6 here.
    np.testing.assert_allclose(rescaled, brute_force, rtol=1e-5)


@pytest.mark.integration
def test_rescaled_shot_noise_reweights_like_waveforms_generated_at_every_node(
    reference_metadata: CatalogMetadata,
) -> None:
    sites = ("source_frame_mass_1", "source_frame_mass_2")
    points = [{**reference_metadata.fiducials, "minimum_mass": 0.9, "mass_width": 1.6}]
    rescaled, brute_force = _rescaled_and_brute_force(
        reference_metadata, sites, points, variance=True
    )

    np.testing.assert_allclose(rescaled, brute_force, rtol=1e-5)


@pytest.fixture
def shot_noise_builder(reference_metadata: CatalogMetadata) -> Any:
    """``observation_time -> variance_fn`` on one reference catalog."""
    data = reference_catalog(reference_metadata, batch_keys(41, 1)[0])
    target = build_population(
        reference_metadata.population.model_name,
        **reference_metadata.population.model_kwargs,
    )

    def build(observation_time: float) -> Any:
        return build_rescaled_shot_noise(
            data,
            reference_metadata,
            population=target,
            frequencies=OBSERVED_WAVEFORM.build().frequencies,
            num_redshift_nodes=NUM_NODES,
            density_sites=(),
            observation_time=observation_time,
        )

    return build


@pytest.mark.integration
def test_rescaled_shot_noise_falls_inversely_with_observation_time(
    reference_metadata: CatalogMetadata, shot_noise_builder: Any
) -> None:
    fiducials = reference_metadata.fiducials
    one_year = np.asarray(shot_noise_builder(1.0)(fiducials))
    four_years = np.asarray(shot_noise_builder(4.0)(fiducials))

    np.testing.assert_allclose(four_years, one_year / 4.0, rtol=1e-14)


@pytest.mark.integration
@pytest.mark.parametrize("hubble_constant", [50.0, 90.0])
def test_rescaled_shot_noise_grows_linearly_with_the_hubble_constant(
    reference_metadata: CatalogMetadata,
    shot_noise_builder: Any,
    hubble_constant: float,
) -> None:
    """The squared kernel goes as :math:`D_H / \\chi^2 \\propto H_0`."""
    fiducials = reference_metadata.fiducials
    variance_fn = shot_noise_builder(OBSERVATION_TIME)
    shifted = np.asarray(variance_fn({**fiducials, "H0": hubble_constant}))
    ratio = shifted / np.asarray(variance_fn(fiducials))

    np.testing.assert_allclose(ratio, hubble_constant / fiducials["H0"], rtol=1e-12)


@pytest.mark.integration
def test_rescaled_log_weights_vanish_at_the_fiducials(
    reference_metadata: CatalogMetadata,
) -> None:
    data = reference_catalog(reference_metadata, batch_keys(41, 1)[0])
    target = build_population(
        reference_metadata.population.model_name,
        **reference_metadata.population.model_kwargs,
    )
    _, log_weights_fn = build_rescaled_spectrum(
        data,
        reference_metadata,
        population=target,
        frequencies=OBSERVED_WAVEFORM.build().frequencies,
        num_redshift_nodes=NUM_NODES,
        density_sites=("source_frame_mass_1", "source_frame_mass_2"),
    )

    np.testing.assert_array_equal(
        np.asarray(log_weights_fn(reference_metadata.fiducials)), np.zeros(6)
    )
    assert (
        np.ptp(
            np.asarray(
                log_weights_fn(
                    {
                        **reference_metadata.fiducials,
                        "minimum_mass": 0.9,
                        "mass_width": 1.6,
                    }
                )
            )
        )
        > 0.0
    )


@pytest.mark.integration
def test_rescaled_spectrum_as_a_jit_argument_traces_once_per_shape(
    reference_metadata: CatalogMetadata,
) -> None:
    """Separately built catalogs of one shape share a compilation."""
    target = build_population(
        reference_metadata.population.model_name,
        **reference_metadata.population.model_kwargs,
    )
    traces: list[None] = []

    def body(fn: Any, params: dict[str, float]) -> jax.Array:
        traces.append(None)
        return fn(params)[0]

    spectrum = jax.jit(body)
    results = []
    for seed in (41, 42):
        data = reference_catalog(reference_metadata, batch_keys(seed, 1)[0])
        fn, _ = build_rescaled_spectrum(
            data,
            reference_metadata,
            population=target,
            frequencies=data["frequencies"][:20],
            num_redshift_nodes=4,
            density_sites=(),
        )
        results.append(np.asarray(spectrum(fn, reference_metadata.fiducials)))

    assert len(traces) == 1
    assert not np.allclose(results[0], results[1], rtol=1e-6, atol=0.0)


@pytest.mark.integration
def test_rescaled_shot_noise_as_a_jit_argument_traces_once_per_shape(
    reference_metadata: CatalogMetadata,
) -> None:
    """Separately built catalogs of one shape share a compilation."""
    target = build_population(
        reference_metadata.population.model_name,
        **reference_metadata.population.model_kwargs,
    )
    traces: list[None] = []

    def body(fn: Any, params: dict[str, float]) -> jax.Array:
        traces.append(None)
        return fn(params)

    variance = jax.jit(body)
    for seed in (41, 42):
        data = reference_catalog(reference_metadata, batch_keys(seed, 1)[0])
        fn = build_rescaled_shot_noise(
            data,
            reference_metadata,
            population=target,
            frequencies=data["frequencies"][:20],
            num_redshift_nodes=4,
            density_sites=(),
            observation_time=OBSERVATION_TIME,
        )
        variance(fn, reference_metadata.fiducials)

    assert len(traces) == 1


@pytest.mark.parametrize("xi_0", [1.0, 0.8])
def test_phinney_kernel_times_the_distance_ratio_squared_is_the_rate_density(
    xi_0: float,
) -> None:
    """Undoing the distance factor leaves ``psi / (1 + z) dV_c/dz`` per second."""
    redshift, weights = redshift_quadrature(0.1, 5.0, 24)
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

    np.testing.assert_allclose(
        np.sum(weights * undone),
        total_merger_rate(redshift, weights, merger_rate, hubble_constant, omega_m),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("parameter", "exponent"), [("H0", -1.0), ("local_merger_rate", 1.0)]
)
@pytest.mark.parametrize("factor", [0.5, 1.3, 2.0])
def test_spectrum_scales_as_a_power_of_h0_and_the_local_merger_rate(
    parameter: str, exponent: float, factor: float
) -> None:
    """``S ~ 1/H0`` and ``S ~ R0``: the exponents the cosmology and kernel imply."""
    spectrum, _ = build_reference_spectrum(*build_reference_catalog(num_sources=16))
    fiducial = FIDUCIALS[parameter]

    baseline, _ = spectrum(FIDUCIALS)
    scaled, _ = spectrum({**FIDUCIALS, parameter: factor * fiducial})

    np.testing.assert_allclose(
        np.asarray(scaled), factor**exponent * np.asarray(baseline), rtol=1e-8
    )
