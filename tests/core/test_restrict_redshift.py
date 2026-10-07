"""Tests for restricting a polarization-power draw to a redshift window."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpyro import handlers

from astrogwb.frequency import bin_widths
from astrogwb.populations import PopulationMetadata
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    PolarizationPowerData,
    restrict_redshift,
)
from astrogwb.waveform import WaveformMetadata

#: The population record every draw carries: the density that drew it.
POPULATION_RECORD: dict[str, Any] = {
    "model_name": "bns_coba",
    "model_kwargs": {
        "mass_model": "uniform",
        "minimum_redshift": 0.0,
        "maximum_redshift": 20.0,
        "n_grid": 256,
    },
    "fiducials": {
        "H0": 67.66,
        "Omega_m": 0.3096,
        "gamma": 1.42,
        "kappa": 4.62,
        "z_peak": 1.84,
        "local_merger_rate": 770.0,
        "minimum_mass": 1.0,
        "mass_width": 1.5,
    },
}


@pytest.fixture
def draw_factory():
    """Build a ``(data, metadata)`` pair on a two-bin grid from redshifts."""

    def _draw(redshift: np.ndarray) -> tuple[PolarizationPowerData, CatalogMetadata]:
        num_samples = redshift.size
        data = PolarizationPowerData(
            frequencies=np.array([10.0, 12.0]),
            polarization_power=np.arange(2 * num_samples, dtype=np.float64).reshape(
                2, num_samples
            ),
            source_parameters={
                "redshift": redshift,
                "luminosity_distance": 1e3 * (1.0 + redshift),
            },
        )
        metadata = CatalogMetadata(
            waveform=WaveformMetadata(
                approximant="FakeWaveform",
                minimum_frequency=10.0,
                maximum_frequency=12.5,
                reference_frequency=11.0,
                sampling_frequency=64.0,
                frequency_resolution=2.0,
            ),
            population=PopulationMetadata(
                model_name=POPULATION_RECORD["model_name"],
                model_kwargs=POPULATION_RECORD["model_kwargs"],
            ),
            fiducials=POPULATION_RECORD["fiducials"],
            num_samples=num_samples,
        )
        return data, metadata

    return _draw


def test_restrict_redshift_narrows_the_samples_and_the_population_together(
    draw_factory,
) -> None:
    """Truncating changes the density's *normalization*, so both must move."""
    data, metadata = draw_factory(np.array([0.1, 0.5, 1.5, 19.0]))
    restricted, narrowed = restrict_redshift(data, metadata, 0.3, 2.0)

    np.testing.assert_array_equal(
        restricted["source_parameters"]["redshift"], np.array([0.5, 1.5])
    )
    np.testing.assert_array_equal(
        restricted["polarization_power"], data["polarization_power"][:, [1, 2]]
    )
    assert narrowed.num_samples == 2
    assert narrowed.population.model_kwargs["minimum_redshift"] == 0.3
    assert narrowed.population.model_kwargs["maximum_redshift"] == 2.0
    # Everything else about the record travels unchanged.
    assert narrowed.fiducials == metadata.fiducials
    assert narrowed.waveform == metadata.waveform


def test_restrict_redshift_reaches_the_rebuilt_population(draw_factory) -> None:
    """The rebuilt population is built from the one flat kwargs mapping it rewrites."""
    data, metadata = draw_factory(np.array([0.1, 0.5, 1.5, 19.0]))
    _, narrowed = restrict_redshift(data, metadata, 0.3, 2.0)

    rate, model = narrowed.population.build()(narrowed.fiducials)
    with handlers.seed(rng_seed=0):
        redshift = model()["redshift"]
    assert float(rate) > 0.0
    assert 0.3 <= float(redshift) <= 2.0


def test_restrict_redshift_leaves_the_inputs_untouched(draw_factory) -> None:
    data, metadata = draw_factory(np.array([0.1, 0.5, 1.5, 19.0]))
    restrict_redshift(data, metadata, 0.3, 2.0)

    assert metadata.num_samples == 4
    assert data["polarization_power"].shape == (2, 4)
    assert metadata.population.model_kwargs["minimum_redshift"] == 0.0


def test_restrict_redshift_leaves_the_frequency_grid_unchanged(draw_factory) -> None:
    data, metadata = draw_factory(np.array([0.1, 0.5, 1.5, 19.0]))
    restricted, _ = restrict_redshift(data, metadata, 0.3, 2.0)

    np.testing.assert_allclose(
        bin_widths(restricted["frequencies"]), bin_widths(data["frequencies"])
    )


def test_restrict_redshift_rejects_a_window_outside_the_generation_support(
    draw_factory,
) -> None:
    data, metadata = draw_factory(np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match="must lie within"):
        restrict_redshift(data, metadata, 0.3, 25.0)


def test_restrict_redshift_rejects_an_empty_window(draw_factory) -> None:
    data, metadata = draw_factory(np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match="no samples"):
        restrict_redshift(data, metadata, 5.0, 10.0)
