"""Tests for the analytic generator and the frequency grid a catalog carries."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from astrogwb.constants import ISCO_ALPHA
from astrogwb.waveform import (
    AnalyticInspiralGenerator,
    WaveformMetadata,
    inspiral_intensity,
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
        inspiral_intensity(generator.frequencies, source_parameters, alpha=ISCO_ALPHA)
    ).T

    np.testing.assert_array_equal(frequencies, generator.frequencies)
    assert isinstance(actual, jax.Array)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "grid_settings",
    [{"frequency_spacing": "log"}],
    ids=["log"],
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
        inspiral_intensity(frequencies, source_parameters, alpha=ISCO_ALPHA)
    ).T

    assert frequencies[0] == 10.0
    assert frequencies[-1] == 100.0
    assert np.ptp(np.diff(np.asarray(frequencies))) > 0.0
    np.testing.assert_array_equal(actual, expected)


# --------------------------------------------------------------------------- #
# bin_widths: derived from the stored grid, not recorded from the descriptor
# --------------------------------------------------------------------------- #
