"""Tests for per-source polarization Stokes parameters and polarizations."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from astrogwb.constants import ISCO_ALPHA
from astrogwb.waveform import (
    AnalyticInspiralGenerator,
    RippleGenerator,
    WaveformMetadata,
    polarization_stokes,
    quadrupolar_polarizations,
    quadrupolar_stokes_factors,
)


def _ripple_sources() -> dict[str, np.ndarray]:
    return {
        "detector_frame_mass_1": np.array([1.4, 1.3, 1.5]),
        "detector_frame_mass_2": np.array([1.3, 1.2, 1.25]),
        "inclination": np.array([0.0, 0.4, 2.0]),
        "luminosity_distance": np.array([100.0, 200.0, 300.0]),
    }


def _ripple_generator(approximant: str) -> RippleGenerator:
    return RippleGenerator(
        WaveformMetadata(
            approximant=approximant,
            sampling_frequency=1024.0,
            minimum_frequency=20.0,
            maximum_frequency=256.0,
            reference_frequency=20.0,
            frequency_resolution=0.5,
        )
    )


@pytest.fixture
def analytic_generator() -> AnalyticInspiralGenerator:
    return AnalyticInspiralGenerator(
        WaveformMetadata(
            approximant="AnalyticInspiral",
            alpha=ISCO_ALPHA,
            sampling_frequency=2048.0,
            reference_frequency=10.0,
            minimum_frequency=10.0,
            maximum_frequency=500.0,
            frequency_resolution=1.0,
        )
    )


@pytest.fixture
def analytic_sources() -> dict[str, np.ndarray]:
    return {
        "source_frame_mass_1": np.array([1.4, 1.3]),
        "source_frame_mass_2": np.array([1.3, 1.2]),
        "redshift": np.array([0.1, 0.5]),
        "luminosity_distance": np.array([400.0, 3000.0]),
        "inclination": np.array([0.3, 1.9]),
    }


@pytest.mark.parametrize(
    ("plus", "cross", "expected"),
    [
        (1.0, 0.0, (1.0, 1.0, 0.0, 0.0)),
        (0.0, 1.0, (1.0, -1.0, 0.0, 0.0)),
        (1.0, 1.0, (2.0, 0.0, 2.0, 0.0)),
        (1.0, 1j, (2.0, 0.0, 0.0, 2.0)),
    ],
    ids=["plus", "cross", "linear-45", "circular"],
)
def test_polarization_stokes_of_pure_states(
    plus: complex, cross: complex, expected: tuple[float, ...]
) -> None:
    stokes = polarization_stokes(jnp.full((1, 1), plus), jnp.full((1, 1), cross))

    assert stokes.shape == (4, 1, 1)
    np.testing.assert_allclose(np.asarray(stokes)[:, 0, 0], expected)


@pytest.mark.parametrize(
    ("inclination", "expected"),
    [
        (0.0, (2.0, 0.0, 0.0, -2.0)),
        (np.pi, (2.0, 0.0, 0.0, 2.0)),
        (np.pi / 2, (0.25, 0.25, 0.0, 0.0)),
    ],
    ids=["face-on", "face-off", "edge-on"],
)
def test_quadrupolar_stokes_factors_at_special_inclinations(
    inclination: float, expected: tuple[float, ...]
) -> None:
    """Face-on and face-off are circular with opposite V, edge-on is linear."""
    factors = quadrupolar_stokes_factors(jnp.cos(jnp.array([inclination])))
    plus, cross = quadrupolar_polarizations(jnp.ones((1, 2)), jnp.array([inclination]))
    stokes = polarization_stokes(plus, cross)

    np.testing.assert_allclose(
        np.asarray(factors)[:, 0],
        np.array(expected),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(stokes)[..., 0],
        np.repeat(np.array(expected)[:, None], 2, 1),
        atol=1e-15,
    )


def test_ripple_polarization_stokes_intensity_is_the_power() -> None:
    generator = _ripple_generator("TaylorF2")
    sources = _ripple_sources()
    plus, cross = generator.generate_polarizations_batch(sources)
    stokes = polarization_stokes(plus, cross)

    np.testing.assert_allclose(
        np.asarray(stokes[0]),
        np.asarray(generator.generate_batch(sources)),
        rtol=1e-12,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "approximant", ["IMRPhenomXAS_NRTidalv3", "IMRPhenomXPHM", "IMRPhenomXHM"]
)
def test_ripple_polarization_stokes_intensity_is_the_power_for_every_kernel(
    approximant: str,
) -> None:
    generator = _ripple_generator(approximant)
    sources = _ripple_sources()
    plus, cross = generator.generate_polarizations_batch(sources)
    stokes = polarization_stokes(plus, cross)

    np.testing.assert_allclose(
        np.asarray(stokes[0]),
        np.asarray(generator.generate_batch(sources)),
        rtol=1e-11,
    )


@pytest.mark.parametrize("approximant", ["TaylorF2", "IMRPhenomXHM"])
def test_one_source_is_fully_polarized(approximant: str) -> None:
    plus, cross = _ripple_generator(approximant).generate_polarizations_batch(
        _ripple_sources()
    )
    stokes = np.asarray(polarization_stokes(plus, cross))
    intensity, linear_q, linear_u, circular = stokes

    np.testing.assert_allclose(
        linear_q**2 + linear_u**2 + circular**2,
        intensity**2,
        rtol=0.0,
        atol=1e-12 * (intensity**2).max(),
    )


def test_analytic_polarization_stokes_intensity_is_the_power(
    analytic_generator: AnalyticInspiralGenerator,
    analytic_sources: dict[str, np.ndarray],
) -> None:
    plus, cross = analytic_generator.generate_polarizations_batch(analytic_sources)
    stokes = np.asarray(polarization_stokes(plus, cross))
    power = np.asarray(analytic_generator.generate_batch(analytic_sources))

    np.testing.assert_allclose(stokes[0], power, rtol=1e-14)
    np.testing.assert_array_equal(stokes[2], 0.0)
