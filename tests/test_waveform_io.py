from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from asgwb.waveform import FrequencyGrid, WaveformPolarizations
from asgwb.waveform.io import dump_waveform_npz, load_waveform_npz


@pytest.fixture
def grid() -> FrequencyGrid:
    return FrequencyGrid(
        duration=4.0,
        sampling_frequency=2048.0,
        minimum_frequency=20.0,
        maximum_frequency=1024.0,
        reference_frequency=50.0,
    )


@pytest.fixture
def polarizations(grid: FrequencyGrid) -> WaveformPolarizations:
    n = len(grid.frequencies)
    plus = np.arange(n, dtype=np.float64).astype(np.complex128)
    cross = (np.arange(n, dtype=np.float64) * 1j).astype(np.complex128)
    return WaveformPolarizations(grid=grid, plus=plus, cross=cross)


def test_dump_and_load_waveform_npz_round_trip(
    tmp_path: Path, polarizations: WaveformPolarizations
):
    path = tmp_path / "waveform.npz"
    parameters = {"mass_1": 1.4, "mass_2": 1.3}

    dump_waveform_npz(path=path, polarizations=polarizations, parameters=parameters)
    loaded_polarizations, loaded_parameters = load_waveform_npz(path=path)

    assert loaded_polarizations.grid == polarizations.grid
    np.testing.assert_allclose(loaded_polarizations.plus, polarizations.plus)
    np.testing.assert_allclose(loaded_polarizations.cross, polarizations.cross)
    assert loaded_parameters == parameters


def test_dump_waveform_npz_rejects_non_npz_extension(
    tmp_path: Path, polarizations: WaveformPolarizations
):
    with pytest.raises(ValueError, match=".npz"):
        dump_waveform_npz(path=tmp_path / "waveform.h5", polarizations=polarizations)


def test_dump_waveform_npz_rejects_non_scalar_parameter(
    tmp_path: Path, polarizations: WaveformPolarizations
):
    invalid_parameters = cast(dict[str, float], {"mass_1": np.array([1.4, 1.5])})
    with pytest.raises(ValueError, match="must be scalar"):
        dump_waveform_npz(
            path=tmp_path / "waveform.npz",
            polarizations=polarizations,
            parameters=invalid_parameters,
        )


def test_load_waveform_npz_rejects_mismatched_grid(
    tmp_path: Path, polarizations: WaveformPolarizations
):
    path = tmp_path / "waveform.npz"
    dump_waveform_npz(path=path, polarizations=polarizations)
    wrong_grid = FrequencyGrid(
        duration=polarizations.grid.duration,
        sampling_frequency=polarizations.grid.sampling_frequency,
        minimum_frequency=polarizations.grid.minimum_frequency,
        maximum_frequency=polarizations.grid.maximum_frequency,
        reference_frequency=polarizations.grid.reference_frequency + 1.0,
    )

    with pytest.raises(ValueError, match="does not match"):
        load_waveform_npz(path=path, grid=wrong_grid)


def test_waveform_polarizations_wrappers_round_trip(
    tmp_path: Path, polarizations: WaveformPolarizations
):
    path = tmp_path / "waveform.npz"
    parameters = {"lambda_1": 400.0}

    polarizations.dump(path=path, parameters=parameters)
    loaded_polarizations, loaded_parameters = WaveformPolarizations.load(path=path)

    assert loaded_polarizations.grid == polarizations.grid
    np.testing.assert_allclose(loaded_polarizations.plus, polarizations.plus)
    np.testing.assert_allclose(loaded_polarizations.cross, polarizations.cross)
    assert loaded_parameters == parameters
