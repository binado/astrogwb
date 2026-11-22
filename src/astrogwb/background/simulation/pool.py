"""Read a physical event pool from a population catalogue."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astrogwb.background.simulation.draw import SKY_COLUMNS
from astrogwb.utils.io import h5py

__all__ = ["read_population_file"]

#: The two ways a catalogue gives the masses the waveform needs.
_MASS_COLUMNS = (
    ("detector_frame_mass_1", "detector_frame_mass_2"),
    ("source_frame_mass_1", "source_frame_mass_2"),
)

_CATALOGUE_SUFFIXES = {".csv": "csv", ".h5": "hdf5", ".hdf5": "hdf5"}


def _as_catalogue(
    data: np.ndarray | dict[str, np.ndarray], *, source: str
) -> dict[str, np.ndarray]:
    """Named 1-D columns of equal length, or ValueError for anything else."""
    if isinstance(data, np.ndarray):
        names = data.dtype.names
        if names is None:
            raise ValueError(f"{source} must hold named columns.")
        catalogue = {name: np.atleast_1d(np.asarray(data[name])) for name in names}
    else:
        catalogue = {
            name: np.atleast_1d(np.asarray(values)) for name, values in data.items()
        }
    if not catalogue:
        raise ValueError(f"{source} holds no columns.")
    if any(column.ndim != 1 for column in catalogue.values()):
        raise ValueError(f"{source} columns must be one-dimensional.")
    if len({column.shape[0] for column in catalogue.values()}) != 1:
        raise ValueError(f"{source} columns have mismatched lengths.")
    return catalogue


def _read_csv_catalogue(path: Path) -> dict[str, np.ndarray]:
    """Header-named CSV columns as 1-D arrays."""
    data = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    )
    return _as_catalogue(data, source=str(path))


def _read_hdf5_catalogue(path: Path, *, dataset: str) -> dict[str, np.ndarray]:
    """The ``dataset`` node as 1-D columns.

    A compound dataset reads directly; a group of datasets reads one column
    per member.
    """
    with h5py.File(path, "r") as file:
        try:
            node = file[dataset]
        except KeyError:
            raise ValueError(f"{path} has no {dataset!r} catalogue.") from None
        if isinstance(node, h5py.Dataset):
            return _as_catalogue(np.atleast_1d(node[()]), source=f"{path}:{dataset}")
        if not isinstance(node, h5py.Group):
            raise TypeError(f"{path}:{dataset} is not a catalogue.")
        columns = {}
        for name, item in node.items():
            if not isinstance(item, h5py.Dataset):
                raise TypeError(f"{path}:{dataset}/{name} is not a dataset.")
            column = np.atleast_1d(item[()])
            if column.ndim != 1:
                raise ValueError(f"{path}:{dataset}/{name} must be one-dimensional.")
            columns[name] = column
        return _as_catalogue(columns, source=f"{path}:{dataset}")


def read_population_file(
    path: str | Path, *, root: str = "source_parameters"
) -> dict[str, NDArray[np.float64]]:
    """Read a population catalogue as an event pool for the simulator.

    The file is an HDF5 or CSV catalogue with canonical gwmock-pop column
    names. HDF5 catalogues are read from the ``root`` node, a compound
    dataset or a group of 1-D datasets. Besides the waveform parameters it
    must hold ``distance`` (the luminosity distance in Mpc, renamed
    ``luminosity_distance`` here), the sky columns ``right_ascension``,
    ``declination`` and ``polarization_angle``, and either the detector-frame
    or the source-frame masses. Every other column is passed through
    unchanged. All columns are returned as float64.

    Parameters
    ----------
    path
        Catalogue path, ``.h5``, ``.hdf5`` or ``.csv``.
    root
        Dataset or group path inside an HDF5 file. Ignored for CSV files.

    Returns
    -------
    dict of str to numpy.ndarray
        One-dimensional columns in this package's names.

    Raises
    ------
    ValueError
        If the file format is unsupported, the catalogue is malformed, or a
        required column is missing.
    TypeError
        If the HDF5 node is neither a dataset nor a group of datasets.
    """
    suffix = Path(path).suffix.lower()
    file_format = _CATALOGUE_SUFFIXES.get(suffix)
    if file_format is None:
        supported = ", ".join(sorted(_CATALOGUE_SUFFIXES))
        raise ValueError(
            f"unsupported catalogue format for {path}: supports {supported}."
        )
    if file_format == "hdf5":
        catalogue = _read_hdf5_catalogue(Path(path), dataset=root)
    else:
        catalogue = _read_csv_catalogue(Path(path))

    required = ["distance", *SKY_COLUMNS]
    missing = [name for name in required if name not in catalogue]
    if missing:
        raise ValueError(f"{path} is missing the columns {missing}")
    if not any(all(name in catalogue for name in pair) for pair in _MASS_COLUMNS):
        raise ValueError(
            f"{path} needs detector_frame_mass_1/2 or source_frame_mass_1/2 columns"
        )
    pool = {
        name: np.asarray(values, dtype=np.float64) for name, values in catalogue.items()
    }
    pool["luminosity_distance"] = pool.pop("distance")
    return pool
