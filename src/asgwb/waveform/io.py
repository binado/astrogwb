from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from .grid import FrequencyGrid
from .polarizations import WaveformPolarizations

_NPZ_SUFFIX = ".npz"
_GRID_KEY_PREFIX = "grid_"
_PARAM_KEY_PREFIX = "param_"
_WAVEFORM_KEYS = ("plus", "cross")
_REQUIRED_GRID_KEYS = (
    f"{_GRID_KEY_PREFIX}duration",
    f"{_GRID_KEY_PREFIX}sampling_frequency",
    f"{_GRID_KEY_PREFIX}minimum_frequency",
    f"{_GRID_KEY_PREFIX}maximum_frequency",
    f"{_GRID_KEY_PREFIX}reference_frequency",
)


def _validate_npz_path(path: Path) -> None:
    if path.suffix.lower() != _NPZ_SUFFIX:
        raise ValueError(f"Expected a '{_NPZ_SUFFIX}' file, got '{path.suffix}'")


def dump_waveform_npz(
    path: Path,
    polarizations: WaveformPolarizations,
    parameters: dict[str, float] | None = None,
) -> None:
    _validate_npz_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    param_payload: dict[str, float] = {}
    if parameters is not None:
        for key, value in parameters.items():
            if not np.isscalar(value):
                raise ValueError(f"Parameter '{key}' must be scalar, got {type(value)}")
            param_payload[f"{_PARAM_KEY_PREFIX}{key}"] = float(value)

    maximum_frequency = polarizations.grid.maximum_frequency
    if maximum_frequency is None:
        raise ValueError("grid.maximum_frequency must be resolved before serialization")

    payload = {
        "plus": polarizations.plus,
        "cross": polarizations.cross,
        "grid_duration": polarizations.grid.duration,
        "grid_sampling_frequency": polarizations.grid.sampling_frequency,
        "grid_minimum_frequency": polarizations.grid.minimum_frequency,
        "grid_maximum_frequency": maximum_frequency,
        "grid_reference_frequency": polarizations.grid.reference_frequency,
        **param_payload,
    }
    np.savez(path, **cast(dict[str, Any], payload))


def _read_grid(data: np.lib.npyio.NpzFile) -> FrequencyGrid:
    missing_keys = [key for key in _REQUIRED_GRID_KEYS if key not in data.files]
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing required NPZ keys: {missing}")

    return FrequencyGrid(
        duration=float(data["grid_duration"].item()),
        sampling_frequency=float(data["grid_sampling_frequency"].item()),
        minimum_frequency=float(data["grid_minimum_frequency"].item()),
        maximum_frequency=float(data["grid_maximum_frequency"].item()),
        reference_frequency=float(data["grid_reference_frequency"].item()),
    )


def _read_polarizations(
    data: np.lib.npyio.NpzFile, grid: FrequencyGrid
) -> WaveformPolarizations:
    if all(key in data.files for key in _WAVEFORM_KEYS):
        return WaveformPolarizations(
            grid=grid,
            plus=np.asarray(data["plus"], dtype=np.complex128),
            cross=np.asarray(data["cross"], dtype=np.complex128),
        )
    missing = [key for key in _WAVEFORM_KEYS if key not in data.files]
    raise ValueError(f"Missing waveform polarization keys in NPZ file: {missing}")


def load_waveform_npz(
    path: Path, grid: FrequencyGrid | None = None
) -> tuple[WaveformPolarizations, dict[str, float]]:
    _validate_npz_path(path)

    with np.load(path) as data:
        serialized_grid = _read_grid(data)
        if grid is not None and serialized_grid != grid:
            raise ValueError(
                "Provided grid does not match serialized grid metadata in NPZ file"
            )

        parameters = {
            key.removeprefix(_PARAM_KEY_PREFIX): float(data[key].item())
            for key in data.files
            if key.startswith(_PARAM_KEY_PREFIX)
        }
        polarizations = _read_polarizations(
            data=data,
            grid=serialized_grid if grid is None else grid,
        )

    return polarizations, parameters
