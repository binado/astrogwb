"""The redshift-node importance catalog, its draw, and the rescaled reference catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from astrogwb.gwb.importance import (
    EFFECTIVE_INCLINATION,
    ImportanceCatalogMetadata,
    build_importance_spectrum,
    build_rescaled_spectrum,
    importance_catalog,
    redshift_quadrature,
    reference_catalog,
)
from astrogwb.populations import PopulationMetadata, build_population
from astrogwb.simulators.core import batch_keys, load, write
from astrogwb.simulators.polarization_power import CatalogMetadata
from astrogwb.waveform import WaveformMetadata
from astrogwb.waveform.generator.analytical import inclination_factor


@pytest.fixture
def metadata() -> ImportanceCatalogMetadata:
    """A tiny closed-form catalog: six draws at three nodes on a four-bin grid."""
    return ImportanceCatalogMetadata(
        waveform=WaveformMetadata(
            approximant="AnalyticInspiral",
            minimum_frequency=20.0,
            maximum_frequency=50.0,
            reference_frequency=20.0,
            sampling_frequency=128.0,
            frequency_resolution=10.0,
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
        num_redshift_nodes=3,
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
def test_importance_catalog_places_every_draw_at_every_node(
    metadata: ImportanceCatalogMetadata,
) -> None:
    data = importance_catalog(metadata, batch_keys(41, 1)[0], chunk_size=4)

    nodes, _ = metadata.nodes()
    columns = data["source_parameters"]
    assert data["polarization_power"].shape == (4, 18)
    np.testing.assert_array_equal(columns["redshift"], np.repeat(nodes, 6))
    np.testing.assert_array_equal(
        columns["source_frame_mass_1"].reshape(3, 6),
        np.tile(columns["source_frame_mass_1"][:6], (3, 1)),
    )
    np.testing.assert_allclose(
        columns["detector_frame_mass_1"],
        columns["source_frame_mass_1"] * (1.0 + columns["redshift"]),
    )


@pytest.mark.integration
def test_importance_catalog_round_trips_through_a_file(
    metadata: ImportanceCatalogMetadata, tmp_path: Path
) -> None:
    fresh = importance_catalog(metadata, batch_keys(41, 1)[0])
    path = write(tmp_path / "catalog.h5", fresh, metadata, seed=41)

    data, recorded, attrs = load(path, ImportanceCatalogMetadata)

    np.testing.assert_array_equal(
        data["polarization_power"], fresh["polarization_power"]
    )
    assert recorded == metadata
    assert attrs["seed"] == 41


@pytest.fixture
def reference_metadata(metadata: ImportanceCatalogMetadata) -> CatalogMetadata:
    """The same draw, on a fine log grid covering every rescaled query.

    Queries reach ``50 Hz * (1 + 5) / (1 + 0.1)``, about 273 Hz; the analytic
    inspiral ends far above that for these masses, so the comparison sees only
    the rescaling and the interpolation.
    """
    return CatalogMetadata(
        waveform=metadata.waveform.model_copy(
            update={
                "frequency_spacing": "log",
                "maximum_frequency": 300.0,
                "frequency_resolution": 0.02,
            }
        ),
        population=metadata.population,
        fiducials=metadata.fiducials,
        num_samples=metadata.num_samples,
    )


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


def _both_spectra(
    metadata: ImportanceCatalogMetadata,
    reference_metadata: CatalogMetadata,
    density_sites: tuple[str, ...],
    points: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Node and rescaled spectra and log weights, on the same draws and nodes."""
    key = batch_keys(41, 1)[0]
    nodes = importance_catalog(metadata, key)
    reference = reference_catalog(reference_metadata, key)
    np.testing.assert_array_equal(
        reference["source_parameters"]["source_frame_mass_1"],
        nodes["source_parameters"]["source_frame_mass_1"][: metadata.num_samples],
    )
    target = build_population(
        metadata.population.model_name, **metadata.population.model_kwargs
    )
    node_fn, node_weights = build_importance_spectrum(
        nodes, metadata, population=target, density_sites=density_sites
    )
    rescaled_fn, rescaled_weights = build_rescaled_spectrum(
        reference,
        reference_metadata,
        population=target,
        frequencies=nodes["frequencies"],
        num_redshift_nodes=metadata.num_redshift_nodes,
        density_sites=density_sites,
    )
    return (
        np.stack([np.asarray(node_fn(p)[0]) for p in points]),
        np.stack([np.asarray(rescaled_fn(p)[0]) for p in points]),
        np.stack([np.asarray(node_weights(p)) for p in points]),
        np.stack([np.asarray(rescaled_weights(p)) for p in points]),
    )


@pytest.mark.integration
def test_rescaled_spectrum_matches_the_node_spectrum_on_the_same_draws(
    metadata: ImportanceCatalogMetadata, reference_metadata: CatalogMetadata
) -> None:
    fiducials = metadata.fiducials
    points = [
        fiducials,
        {**fiducials, "H0": 60.0},
        {**fiducials, "xi_0": 1.2, "xi_n": 2.0},
    ]
    node, rescaled, _, _ = _both_spectra(metadata, reference_metadata, (), points)

    # Linear interpolation of an f^(-7/3) power law on a 0.1% log grid.
    np.testing.assert_allclose(rescaled, node, rtol=1e-5)


@pytest.mark.integration
def test_rescaled_spectrum_reweights_like_the_node_spectrum(
    metadata: ImportanceCatalogMetadata, reference_metadata: CatalogMetadata
) -> None:
    sites = ("source_frame_mass_1", "source_frame_mass_2")
    points = [{**metadata.fiducials, "mass_width": 1.8}]
    node, rescaled, node_weights, rescaled_weights = _both_spectra(
        metadata, reference_metadata, sites, points
    )

    assert np.ptp(rescaled_weights) > 0.0
    np.testing.assert_allclose(rescaled_weights, node_weights, rtol=1e-12)
    np.testing.assert_allclose(rescaled, node, rtol=1e-5)


@pytest.mark.integration
def test_rescaled_spectrum_as_a_jit_argument_traces_once_per_shape(
    metadata: ImportanceCatalogMetadata, reference_metadata: CatalogMetadata
) -> None:
    """Separately built catalogs of one shape share a compilation."""
    target = build_population(
        metadata.population.model_name, **metadata.population.model_kwargs
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
        results.append(np.asarray(spectrum(fn, metadata.fiducials)))

    assert len(traces) == 1
    assert not np.allclose(results[0], results[1], rtol=1e-6, atol=0.0)
