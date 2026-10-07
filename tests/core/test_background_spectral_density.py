"""Composition of background spectral densities along their draw axis."""

from __future__ import annotations

import numpy as np
import pytest

from astrogwb.simulators.spectra import (
    BackgroundSpectralDensityData,
    stack_spectra,
)


def _draws(num_draws: int) -> BackgroundSpectralDensityData:
    return {
        "frequencies": np.array([20.0, 30.0, 40.0, 50.0]),
        "spectral_density": np.arange(num_draws * 4, dtype=np.float64).reshape(
            num_draws, 4
        ),
        "n_events": np.full(num_draws, 4, dtype=np.int64),
        "total_merger_rate": np.arange(num_draws, dtype=np.float64) + 1,
        "hyperparameters": {
            "H0": np.full(num_draws, 67.66),
            "local_merger_rate": np.linspace(710.0, 790.0, num_draws),
        },
    }


@pytest.mark.parametrize("sizes", [(1,), (1, 1, 1), (2, 1, 3)])
def test_stack_spectra_concatenates_the_existing_draw_axis(
    sizes: tuple[int, ...],
) -> None:
    parts = [_draws(size) for size in sizes]

    combined = stack_spectra(parts)

    assert combined["frequencies"] is parts[0]["frequencies"]
    assert combined["spectral_density"].shape == (sum(sizes), 4)
    offset = 0
    for part, size in zip(parts, sizes, strict=True):
        rows = slice(offset, offset + size)
        np.testing.assert_array_equal(
            combined["spectral_density"][rows], part["spectral_density"]
        )
        np.testing.assert_array_equal(combined["n_events"][rows], part["n_events"])
        np.testing.assert_array_equal(
            combined["total_merger_rate"][rows], part["total_merger_rate"]
        )
        for name, values in part["hyperparameters"].items():
            np.testing.assert_array_equal(
                combined["hyperparameters"][name][rows], values
            )
        offset += size
