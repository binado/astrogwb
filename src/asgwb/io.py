from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, overload

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from .waveform import FrequencyGrid

if TYPE_CHECKING:
    from .gwb import SpectralDensity
    from pandas.io.parsers import TextFileReader

_HDF5_SUFFIXES = frozenset({".h5", ".hdf5"})
_GRID_KEY_PREFIX = "grid_"
_SPECTRAL_DENSITY_DATASET = "spectral_density"
_REQUIRED_GRID_KEYS = (
    f"{_GRID_KEY_PREFIX}duration",
    f"{_GRID_KEY_PREFIX}sampling_frequency",
    f"{_GRID_KEY_PREFIX}minimum_frequency",
    f"{_GRID_KEY_PREFIX}maximum_frequency",
    f"{_GRID_KEY_PREFIX}reference_frequency",
)


@overload
def load_injection_file(
    path: Path, iterator: bool = False, **kwargs
) -> pd.DataFrame: ...


@overload
def load_injection_file(
    path: Path, iterator: bool = True, **kwargs
) -> TextFileReader: ...


def load_injection_file(
    path: Path, iterator: bool, chunksize: int | None = None, **kwargs
) -> pd.DataFrame | TextFileReader:
    csv_kwargs = kwargs.copy()
    csv_kwargs.update(
        {
            "sep": " ",
            "header": 0,
            "engine": "c",
            "iterator": iterator,
            "chunksize": chunksize,
        }
    )
    return pd.read_csv(path, **csv_kwargs)


def _validate_spectral_density_path(path: Path) -> None:
    if path.suffix.lower() not in _HDF5_SUFFIXES:
        allowed_suffixes = ", ".join(sorted(_HDF5_SUFFIXES))
        raise ValueError(
            f"Expected a spectral density file with one of {allowed_suffixes}, "
            f"got '{path.suffix}'"
        )


def dump_spectral_density_hdf5(path: Path, spectral_density: SpectralDensity) -> None:
    _validate_spectral_density_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    maximum_frequency = spectral_density.grid.maximum_frequency
    if maximum_frequency is None:
        raise ValueError("grid.maximum_frequency must be resolved before serialization")

    with h5py.File(path, "w") as hf:
        hf.create_dataset(
            _SPECTRAL_DENSITY_DATASET,
            data=spectral_density.spectral_density,
            dtype=np.float64,
            compression="gzip",
        )
        hf.attrs[f"{_GRID_KEY_PREFIX}duration"] = spectral_density.grid.duration
        hf.attrs[f"{_GRID_KEY_PREFIX}sampling_frequency"] = (
            spectral_density.grid.sampling_frequency
        )
        hf.attrs[f"{_GRID_KEY_PREFIX}minimum_frequency"] = (
            spectral_density.grid.minimum_frequency
        )
        hf.attrs[f"{_GRID_KEY_PREFIX}maximum_frequency"] = maximum_frequency
        hf.attrs[f"{_GRID_KEY_PREFIX}reference_frequency"] = (
            spectral_density.grid.reference_frequency
        )


def _read_spectral_density_grid(hf: h5py.File) -> FrequencyGrid:
    missing_keys = [key for key in _REQUIRED_GRID_KEYS if key not in hf.attrs]
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing required HDF5 attributes: {missing}")

    return FrequencyGrid(
        duration=float(hf.attrs[f"{_GRID_KEY_PREFIX}duration"]),
        sampling_frequency=float(hf.attrs[f"{_GRID_KEY_PREFIX}sampling_frequency"]),
        minimum_frequency=float(hf.attrs[f"{_GRID_KEY_PREFIX}minimum_frequency"]),
        maximum_frequency=float(hf.attrs[f"{_GRID_KEY_PREFIX}maximum_frequency"]),
        reference_frequency=float(hf.attrs[f"{_GRID_KEY_PREFIX}reference_frequency"]),
    )


def _read_spectral_density_dataset(hf: h5py.File) -> npt.NDArray[np.float64]:
    if _SPECTRAL_DENSITY_DATASET not in hf:
        raise ValueError(
            f"Missing required dataset '/{_SPECTRAL_DENSITY_DATASET}' in {hf.filename}"
        )

    dataset = hf[_SPECTRAL_DENSITY_DATASET]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(
            f"Expected '/{_SPECTRAL_DENSITY_DATASET}' to be an h5py.Dataset, "
            f"got {type(dataset).__name__}. This may indicate file corruption or "
            f"that the HDF5 group was accessed instead of a dataset."
        )
    spectral_density = np.asarray(dataset[:], dtype=np.float64)
    if spectral_density.ndim != 1:
        raise ValueError(
            "spectral_density dataset must be 1D, "
            f"got {spectral_density.ndim} dimensions"
        )
    return spectral_density


def load_spectral_density_hdf5(path: Path) -> SpectralDensity:
    _validate_spectral_density_path(path)

    with h5py.File(path, "r") as hf:
        grid = _read_spectral_density_grid(hf)
        spectral_density = _read_spectral_density_dataset(hf)

    from .gwb import SpectralDensity

    return SpectralDensity(grid=grid, spectral_density=spectral_density)
