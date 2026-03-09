from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asgwb.gwb import SpectralDensity
from asgwb.waveform import FrequencyGrid


def test_spectral_density_roundtrip(tmp_path: Path) -> None:
    grid = FrequencyGrid(
        duration=1.0,
        sampling_frequency=8.0,
        reference_frequency=20.0,
        minimum_frequency=0.0,
        maximum_frequency=4.0,
    )
    spectral_density = SpectralDensity(
        grid=grid,
        spectral_density=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
    )

    path = tmp_path / "spectral_density.h5"
    spectral_density.dump(path)
    loaded = SpectralDensity.load(path)

    assert loaded.grid == spectral_density.grid
    np.testing.assert_allclose(
        loaded.spectral_density, spectral_density.spectral_density
    )


def test_spectral_density_rejects_shape_mismatch() -> None:
    grid = FrequencyGrid(
        duration=1.0,
        sampling_frequency=8.0,
        reference_frequency=20.0,
        minimum_frequency=0.0,
        maximum_frequency=4.0,
    )

    with pytest.raises(ValueError, match="spectral_density shape does not match"):
        SpectralDensity(
            grid=grid,
            spectral_density=np.array([1.0, 2.0], dtype=np.float64),
        )
