"""The rescaled reference-redshift spectrum, its catalog, and its quadrature."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
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
from astrogwb.importance.catalog import (
    CatalogMetadata,
    polarization_power_data,
    read_catalog,
)
from astrogwb.importance.spectrum import (
    INCLINATION_MEAN,
    INCLINATION_RAW_SECOND_MOMENT,
    INCLINATION_SECOND_MOMENT,
    REFERENCE_INCLINATION,
    _pin_redshift_and_inclination,
    phinney_kernel,
    redshift_quadrature,
    reference_catalog,
)
from astrogwb.inference import (
    ImportanceCompoundPoissonLikelihood,
    ImportanceGaussianLikelihood,
)
from astrogwb.populations import BNSCobaPopulation, joint_model
from astrogwb.simulators import realization_key
from astrogwb.utils import gauss_legendre_nodes_weights
from astrogwb.utils.io import write
from astrogwb.utils.numpyro import sample_and_logprob
from astrogwb.waveform import WaveformMetadata, inclination_factor, waveform_generator

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
        population=BNSCobaPopulation(
            mass_model="uniform",
            minimum_redshift=0.1,
            maximum_redshift=5.0,
            # Fine: the brute force reads the table, the spectrum the closed
            # form, and they agree only to the table's interpolation error.
            n_grid=8192,
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


def test_reference_inclination_is_face_on() -> None:
    assert float(inclination_factor(REFERENCE_INCLINATION)) == pytest.approx(
        2.0, rel=1e-14
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


@pytest.mark.integration
def test_sobol_reference_catalog_stratifies_the_primary_mass(
    reference_metadata: CatalogMetadata,
) -> None:
    """Pinned sites take no coordinate, so the primary mass takes the first."""
    metadata = CatalogMetadata.model_validate(
        {**reference_metadata.model_dump(), "num_samples": 8, "sampling": "sobol"}
    )
    data = reference_catalog(metadata, realization_key(41, 0))

    columns = data["source_parameters"]
    np.testing.assert_array_equal(columns["redshift"], np.full(8, 0.1))
    np.testing.assert_array_equal(
        columns["inclination"], np.full(8, REFERENCE_INCLINATION)
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
    population = metadata.population
    minimum, maximum = population.minimum_redshift, population.maximum_redshift
    redshift, weights = redshift_quadrature(minimum, maximum, NUM_NODES)
    fiducial_model = joint_model(*population.build()(metadata.fiducials))
    samples = sample_and_logprob(
        _pin_redshift_and_inclination(fiducial_model, minimum),
        key,
        num_samples=metadata.num_samples,
        density_sites=(),
    )[0]
    # Every source at every node, node-major: row ``j * N + k`` is source ``k``
    # at node ``j``, replayed through the model so redshift-derived columns
    # (the luminosity distance) are recomputed.
    count = metadata.num_samples
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
            polarization_power_data(
                generator,
                {**rows, "inclination": np.full(count * redshift.size, np.arccos(c))},
            )["polarization_power"]
            for c in cosine
        ]
    ).reshape(5, -1, NUM_NODES, count)
    power = np.einsum("l,lfzn->fzn", orientation_weights / 2, powers)
    squared_power = np.einsum("l,lfzn->fzn", orientation_weights / 2, powers**2)

    redshift_distribution, source_model = population.build()(params)
    merger_rate = redshift_distribution.total_merger_rate()
    model = joint_model(redshift_distribution, source_model)
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
    metadata: CatalogMetadata,
    density_sites: tuple[str, ...],
    points: list[dict[str, float]],
    *,
    variance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rescaled and brute-force spectra, or variances, on the same draws and nodes."""
    key = realization_key(41, 0)
    reference = reference_catalog(metadata, key)
    target = metadata.population.build()
    settings: dict[str, Any] = {
        "population": target,
        "frequencies": waveform_generator(OBSERVED_WAVEFORM).frequencies,
        "num_redshift_nodes": NUM_NODES,
        "density_sites": density_sites,
    }
    likelihood = ImportanceGaussianLikelihood.from_catalog(
        reference,
        metadata,
        shot_noise="amplitude" if variance else None,
        observation_time=OBSERVATION_TIME if variance else None,
        **settings,
    )
    rescaled = [np.asarray(likelihood.predict(p)[int(variance)]) for p in points]
    brute_force = [
        _spectrum_on_nodes(metadata, key, p, density_sites)[int(variance)]
        for p in points
    ]
    return np.stack(rescaled), np.stack(brute_force)


@pytest.mark.integration
@pytest.mark.parametrize("variance", [False, True], ids=["spectrum", "shot-noise"])
def test_rescaled_prediction_matches_waveforms_generated_at_every_node(
    reference_metadata: CatalogMetadata, variance: bool
) -> None:
    fiducials = reference_metadata.fiducials
    points = [
        fiducials,
        {**fiducials, "H0": 60.0},
        {**fiducials, "xi_0": 1.2, "xi_n": 2.0},
    ]
    rescaled, brute_force = _rescaled_and_brute_force(
        reference_metadata, (), points, variance=variance
    )

    # Linear interpolation of an f^(-7/3) power law on a 0.1% log grid.
    np.testing.assert_allclose(rescaled, brute_force, rtol=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize("variance", [False, True], ids=["spectrum", "shot-noise"])
def test_rescaled_prediction_reweights_like_waveforms_generated_at_every_node(
    reference_metadata: CatalogMetadata, variance: bool
) -> None:
    sites = ("source_frame_mass_1", "source_frame_mass_2")
    points = [{**reference_metadata.fiducials, "minimum_mass": 0.9, "mass_width": 1.6}]
    rescaled, brute_force = _rescaled_and_brute_force(
        reference_metadata, sites, points, variance=variance
    )

    np.testing.assert_allclose(rescaled, brute_force, rtol=1e-5)


@pytest.fixture
def shot_noise_builder(reference_metadata: CatalogMetadata) -> Any:
    """``observation_time -> likelihood`` on one reference catalog."""
    data = reference_catalog(reference_metadata, realization_key(41, 0))
    target = reference_metadata.population.build()

    def build(observation_time: float) -> ImportanceGaussianLikelihood:
        return ImportanceGaussianLikelihood.from_catalog(
            data,
            reference_metadata,
            population=target,
            frequencies=np.asarray(waveform_generator(OBSERVED_WAVEFORM).frequencies),
            num_redshift_nodes=NUM_NODES,
            density_sites=(),
            shot_noise="amplitude",
            observation_time=observation_time,
        )

    return build


@pytest.mark.integration
def test_rescaled_shot_noise_grows_linearly_with_the_hubble_constant(
    reference_metadata: CatalogMetadata,
    shot_noise_builder: Any,
) -> None:
    """The squared kernel goes as :math:`D_H / \\chi^2 \\propto H_0`."""
    hubble_constant = 90.0
    fiducials = reference_metadata.fiducials
    likelihood = shot_noise_builder(OBSERVATION_TIME)
    shifted = np.asarray(likelihood.predict({**fiducials, "H0": hubble_constant})[1])
    ratio = shifted / np.asarray(likelihood.predict(fiducials)[1])

    np.testing.assert_allclose(ratio, hubble_constant / fiducials["H0"], rtol=1e-12)


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
def test_spectrum_scales_as_a_power_of_h0_and_the_local_merger_rate(
    parameter: str, exponent: float
) -> None:
    """``S ~ 1/H0`` and ``S ~ R0``: the exponents the cosmology and kernel imply."""
    spectrum = build_reference_spectrum(
        *build_reference_catalog(num_sources=16)
    ).spectrum
    fiducial = FIDUCIALS[parameter]
    factor = 1.3

    baseline = spectrum(FIDUCIALS)
    scaled = spectrum({**FIDUCIALS, parameter: factor * fiducial})

    np.testing.assert_allclose(
        np.asarray(scaled), factor**exponent * np.asarray(baseline), rtol=1e-8
    )


def _at_window_edge(metadata: CatalogMetadata, edge: float) -> CatalogMetadata:
    """``metadata`` with the population's window starting at ``edge``."""
    return metadata.model_copy(
        update={"population": metadata.population.replace(minimum_redshift=edge)}
    )


def _bound_predictions(
    data: Any,
    metadata: CatalogMetadata,
    *,
    minimum_redshift: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The spectrum and shot-noise variance of ``data`` at the fiducials."""
    likelihood = ImportanceGaussianLikelihood.from_catalog(
        data,
        metadata,
        population=metadata.population.build(),
        frequencies=np.asarray(waveform_generator(OBSERVED_WAVEFORM).frequencies),
        num_redshift_nodes=NUM_NODES,
        density_sites=(),
        shot_noise="amplitude",
        observation_time=OBSERVATION_TIME,
        minimum_redshift=minimum_redshift,
    )
    mean, variance, _ = likelihood.predict(metadata.fiducials)
    return np.asarray(mean), np.asarray(variance)


@pytest.mark.integration
@pytest.mark.parametrize("window_edge", [0.5, 1.5])
def test_low_reference_catalog_matches_one_generated_at_the_window_edge(
    reference_metadata: CatalogMetadata, window_edge: float
) -> None:
    """Same key, hence same intrinsic draws: only the interpolation differs."""
    key = realization_key(41, 0)
    low = reference_catalog(reference_metadata, key)
    direct_metadata = _at_window_edge(reference_metadata, window_edge)
    direct = reference_catalog(direct_metadata, key)
    for name in ("source_frame_mass_1", "source_frame_mass_2"):
        np.testing.assert_array_equal(
            low["source_parameters"][name], direct["source_parameters"][name]
        )

    rescaled = _bound_predictions(low, reference_metadata, minimum_redshift=window_edge)
    expected = _bound_predictions(direct, direct_metadata)

    # Linear interpolation of an f^(-7/3) power law on a 0.1% log grid, as in
    # the comparisons against waveforms generated at every node.
    np.testing.assert_allclose(rescaled[0], expected[0], rtol=1e-5)
    np.testing.assert_allclose(rescaled[1], expected[1], rtol=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize("reference", [0.02, 0.05, 0.1])
def test_reference_catalog_distance_is_at_the_reference_redshift(
    reference_metadata: CatalogMetadata, reference: float
) -> None:
    """Below the population's minimum redshift the distance must not be clamped."""
    metadata = reference_metadata.model_copy(update={"reference_redshift": reference})
    columns = reference_catalog(metadata, realization_key(41, 0))["source_parameters"]
    fiducials = metadata.fiducials

    expected = luminosity_distance(reference, fiducials["H0"], fiducials["Omega_m"])
    np.testing.assert_allclose(columns["redshift"], reference)
    np.testing.assert_allclose(
        columns["luminosity_distance"], float(expected), rtol=1e-6
    )


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
def test_face_on_and_legacy_waveforms_give_equivalent_moments(
    reference_metadata: CatalogMetadata, moment: int, approximant: str
) -> None:
    reference_metadata = reference_metadata.model_copy(
        update={
            "waveform": reference_metadata.waveform.model_copy(
                update={"approximant": approximant}
            )
        }
    )
    data = reference_catalog(reference_metadata, realization_key(41, 0))
    legacy_inclination = float(np.arccos(np.sqrt(np.sqrt(11.2) - 3.0)))
    columns = {
        **data["source_parameters"],
        "inclination": np.full(reference_metadata.num_samples, legacy_inclination),
    }
    legacy = polarization_power_data(
        waveform_generator(reference_metadata.waveform), columns
    )
    old_factor = 1.0 if moment == 1 else INCLINATION_SECOND_MOMENT
    new_factor = INCLINATION_MEAN if moment == 1 else INCLINATION_RAW_SECOND_MOMENT
    np.testing.assert_allclose(
        new_factor * data["polarization_power"] ** moment,
        old_factor * legacy["polarization_power"] ** moment,
        atol=0,
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_serialized_reference_convention_is_unambiguous(
    reference_metadata: CatalogMetadata, tmp_path: Path, legacy: bool
) -> None:
    # Serialization provenance must distinguish old files even when their
    # package version matches; defaults on new requests must not mask it.
    path = write(tmp_path / "catalog.h5", {}, reference_metadata, seed=41)
    if legacy:
        with h5py.File(path, "r+") as file:
            file.attrs["metadata"] = reference_metadata.model_dump_json(
                exclude={"reference_inclination"}
            )
    data, metadata, _ = read_catalog(path)
    if legacy:
        assert metadata.reference_inclination is None
        with pytest.raises(ValueError, match="regenerate legacy"):
            ImportanceGaussianLikelihood.from_catalog(
                {**data, "polarization_power": np.ones((1, 1))},
                metadata,
                population=metadata.population.build(),
                frequencies=np.array([20.0]),
                num_redshift_nodes=3,
                density_sites=(),
            )
    else:
        assert metadata == reference_metadata
        assert metadata.reference_inclination == 0.0


@pytest.mark.parametrize("approximant", ["IMRPhenomHM", "IMRPhenomXP_NRTidalv3"])
def test_reference_conversion_rejects_higher_modes_and_precession(
    reference_metadata: CatalogMetadata, approximant: str
) -> None:
    metadata = reference_metadata.model_copy(
        update={
            "waveform": reference_metadata.waveform.model_copy(
                update={"approximant": approximant}
            )
        }
    )
    with pytest.raises(ValueError, match="dominant-quadrupole"):
        reference_catalog(metadata, realization_key(41, 0))


@pytest.mark.integration
@pytest.mark.parametrize(
    "density_sites", [(), ("source_frame_mass_1", "source_frame_mass_2")]
)
def test_dense_covariance_matches_physical_inclination_quadrature(
    reference_metadata: CatalogMetadata, density_sites: tuple[str, ...]
) -> None:
    key = realization_key(41, 0)
    params = {**reference_metadata.fiducials, "minimum_mass": 0.9, "mass_width": 1.6}
    likelihood = ImportanceGaussianLikelihood.from_catalog(
        reference_catalog(reference_metadata, key),
        reference_metadata,
        population=reference_metadata.population.build(),
        frequencies=np.asarray(waveform_generator(OBSERVED_WAVEFORM).frequencies),
        num_redshift_nodes=NUM_NODES,
        density_sites=density_sites,
        observation_time=OBSERVATION_TIME,
        shot_noise="modes",
    )
    _, expected = _spectrum_on_nodes(
        reference_metadata, key, params, density_sites, covariance=True
    )
    # The reference power uses log-frequency interpolation; the oracle generates
    # each waveform at its actual redshift. This bound covers that grid error.
    np.testing.assert_allclose(
        likelihood.catalog_covariance(params), expected, rtol=1e-5, atol=0
    )


@pytest.mark.parametrize("inclination", [None, 1.0])
@pytest.mark.parametrize(
    "likelihood_class",
    [ImportanceGaussianLikelihood, ImportanceCompoundPoissonLikelihood],
)
def test_reference_likelihood_rejects_non_face_on_source_columns(
    reference_metadata: CatalogMetadata,
    inclination: float | None,
    likelihood_class: type[
        ImportanceGaussianLikelihood | ImportanceCompoundPoissonLikelihood
    ],
) -> None:
    columns = {} if inclination is None else {"inclination": np.full(6, inclination)}
    with pytest.raises(ValueError, match="inclinations must all be face-on"):
        likelihood_class.from_catalog(
            {
                "frequencies": np.array([20.0]),
                "polarization_power": np.ones((1, 6)),
                "source_parameters": columns,
            },
            reference_metadata,
            population=reference_metadata.population.build(),
            frequencies=np.array([20.0]),
            num_redshift_nodes=3,
            density_sites=(),
            observation_time=1.0,
        )
