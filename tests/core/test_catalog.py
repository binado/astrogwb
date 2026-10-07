"""Tests for the analytic generator and the frequency grid a catalog carries."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from astrogwb.constants import ISCO_ALPHA
from astrogwb.frequency import bin_widths, uniform_frequency_grid
from astrogwb.waveform import (
    AnalyticInspiralGenerator,
    WaveformMetadata,
    inspiral_polarization_power,
)


@pytest.fixture
def source_parameters() -> dict[str, np.ndarray]:
    return {
        "source_frame_mass_1": np.array([1.4, 1.2]),
        "source_frame_mass_2": np.array([1.3, 1.1]),
        "redshift": np.array([0.1, 0.2]),
        "luminosity_distance": np.array([400.0, 900.0]),
        "inclination": np.array([0.0, 1.0]),
        "integer_label": np.array([7, 8], dtype=np.int16),
    }


@pytest.mark.parametrize(
    ("maximum_frequency", "expected"),
    [
        (20.0, np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])),
        (19.0, np.array([10.0, 12.0, 14.0, 16.0, 18.0])),
    ],
)
def test_generator_includes_largest_in_band_bin(
    maximum_frequency: float, expected: np.ndarray
) -> None:
    generator = WaveformMetadata(
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=maximum_frequency,
        reference_frequency=10.0,
        sampling_frequency=64.0,
        frequency_resolution=2.0,
    )

    np.testing.assert_array_equal(
        uniform_frequency_grid(
            generator.minimum_frequency,
            generator.maximum_frequency,
            generator.frequency_resolution,
        ),
        expected,
    )
    assert generator.maximum_frequency == maximum_frequency


def test_analytic_generator_frequencies_follow_the_metadata_grid() -> None:
    generator = AnalyticInspiralGenerator(
        WaveformMetadata(
            alpha=ISCO_ALPHA,
            approximant="AnalyticInspiral",
            minimum_frequency=10.0,
            maximum_frequency=12.0,
            reference_frequency=10.0,
            sampling_frequency=32.0,
            frequency_resolution=2.0,
        )
    )

    np.testing.assert_array_equal(generator.frequencies, [10.0, 12.0])


def test_analytic_generator_evaluates_on_exact_metadata_grid(
    source_parameters: dict[str, np.ndarray],
) -> None:
    generator = AnalyticInspiralGenerator(
        WaveformMetadata(
            alpha=ISCO_ALPHA,
            approximant="AnalyticInspiral",
            minimum_frequency=9.5,
            maximum_frequency=14.5,
            reference_frequency=10.0,
            sampling_frequency=32.0,
            frequency_resolution=2.0,
        )
    )

    frequencies, actual = generator(source_parameters)
    expected = np.asarray(
        inspiral_polarization_power(
            generator.frequencies, source_parameters, alpha=ISCO_ALPHA
        )
    ).T

    np.testing.assert_array_equal(frequencies, generator.frequencies)
    assert isinstance(actual, jax.Array)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "grid_settings",
    [
        {"frequency_spacing": "loglinear", "turnover_frequency": 12.0},
        {"frequency_spacing": "log"},
    ],
    ids=["loglinear", "log"],
)
def test_analytic_generator_evaluates_on_a_log_spaced_grid(
    source_parameters: dict[str, np.ndarray], grid_settings: dict[str, Any]
) -> None:
    generator = AnalyticInspiralGenerator(
        WaveformMetadata.model_validate(
            {
                "alpha": ISCO_ALPHA,
                "approximant": "AnalyticInspiral",
                "minimum_frequency": 10.0,
                "maximum_frequency": 100.0,
                "reference_frequency": 10.0,
                "sampling_frequency": 256.0,
                "frequency_resolution": 1.0,
                **grid_settings,
            }
        )
    )

    frequencies, actual = generator(source_parameters)
    expected = np.asarray(
        inspiral_polarization_power(frequencies, source_parameters, alpha=ISCO_ALPHA)
    ).T

    assert frequencies[0] == 10.0
    assert frequencies[-1] == 100.0
    assert np.ptp(np.diff(np.asarray(frequencies))) > 0.0
    np.testing.assert_array_equal(actual, expected)


# --------------------------------------------------------------------------- #
# bin_widths: derived from the stored grid, not recorded from the descriptor
# --------------------------------------------------------------------------- #
def test_bin_widths_match_the_generators_requested_resolution() -> None:
    generator = AnalyticInspiralGenerator(
        WaveformMetadata(
            alpha=ISCO_ALPHA,
            approximant="AnalyticInspiral",
            minimum_frequency=10.0,
            maximum_frequency=14.0,
            reference_frequency=10.0,
            sampling_frequency=32.0,
            frequency_resolution=2.0,
        )
    )

    np.testing.assert_allclose(bin_widths(generator.frequencies), 2.0)


def test_bin_widths_of_a_non_uniform_grid_follow_the_local_spacing() -> None:
    np.testing.assert_allclose(
        bin_widths(np.array([10.0, 12.0, 15.0])), [2.0, 2.5, 3.0]
    )


def test_bin_widths_of_a_one_bin_grid_has_no_answer() -> None:
    with pytest.raises(ValueError, match="at least two bins"):
        bin_widths(np.array([10.0]))
