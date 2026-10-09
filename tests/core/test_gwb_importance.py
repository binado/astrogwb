"""The rescaled reference-redshift spectrum, its catalog, and its quadrature."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from astrogwb.gwb.importance import (
    EFFECTIVE_INCLINATION,
    _pin_redshift_and_inclination,
    build_rescaled_spectrum,
    redshift_quadrature,
    reference_catalog,
)
from astrogwb.populations import PopulationMetadata, build_population
from astrogwb.populations.evaluation import evaluate_sources, sample_sources
from astrogwb.simulators.core import batch_keys
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    polarization_power_data,
)
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
                "n_grid": 64,
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


def _spectrum_on_nodes(
    metadata: CatalogMetadata,
    key: jax.Array,
    params: dict[str, float],
    density_sites: tuple[str, ...],
) -> np.ndarray:
    """The quadrature by brute force: every draw generated at every node.

    No rescaling: the waveform is generated in the observer frame at each node,
    at the fiducial distance, and the redshift kernel is read off the target's
    own model row by row.
    """
    population = metadata.population
    minimum, maximum = (
        float(population.model_kwargs[name])
        for name in ("minimum_redshift", "maximum_redshift")
    )
    redshift, weights = redshift_quadrature(minimum, maximum, NUM_NODES)
    _, fiducial_model = population.build()(metadata.fiducials)
    samples = sample_sources(
        _pin_redshift_and_inclination(fiducial_model, minimum),
        key,
        num_samples=metadata.num_samples,
    )
    # Every source at every node, node-major: row ``j * N + k`` is source ``k``
    # at node ``j``, replayed through the model so redshift-derived columns
    # (luminosity distance, detector-frame masses) are recomputed.
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

    merger_rate, model = population.build()(params)
    log_density, outputs = evaluate_sources(model, rows, density_sites=("redshift",))
    distance_ratio = rows["luminosity_distance"] / outputs["luminosity_distance"]
    kernel = weights * np.exp(log_density[::6]) * distance_ratio[::6] ** 2
    first = {name: values[:6] for name, values in rows.items()}
    target, _ = evaluate_sources(model, first, density_sites=density_sites)
    drawn, _ = evaluate_sources(fiducial_model, first, density_sites=density_sites)
    importance = np.exp(np.asarray(target) - np.asarray(drawn))
    return float(merger_rate) * np.einsum("fzn,z,n->f", power, kernel, importance) / 6


def _rescaled_and_brute_force(
    metadata: CatalogMetadata,
    density_sites: tuple[str, ...],
    points: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Rescaled and brute-force spectra on the same draws and nodes."""
    key = batch_keys(41, 1)[0]
    reference = reference_catalog(metadata, key)
    target = build_population(
        metadata.population.model_name, **metadata.population.model_kwargs
    )
    rescaled_fn, _ = build_rescaled_spectrum(
        reference,
        metadata,
        population=target,
        frequencies=OBSERVED_WAVEFORM.build().frequencies,
        num_redshift_nodes=NUM_NODES,
        density_sites=density_sites,
    )
    return (
        np.stack([np.asarray(rescaled_fn(p)[0]) for p in points]),
        np.stack([_spectrum_on_nodes(metadata, key, p, density_sites) for p in points]),
    )


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
    points = [{**reference_metadata.fiducials, "mass_width": 1.8}]
    rescaled, brute_force = _rescaled_and_brute_force(reference_metadata, sites, points)

    np.testing.assert_allclose(rescaled, brute_force, rtol=1e-5)


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
                log_weights_fn({**reference_metadata.fiducials, "mass_width": 1.8})
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
