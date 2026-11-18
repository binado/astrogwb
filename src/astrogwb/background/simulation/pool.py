"""Read a physical event pool written in the gwmock-pop catalogue convention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from astrogwb.background.simulation.simulate import SKY_COLUMNS

__all__ = ["read_event_pool"]

#: The two ways a catalogue gives the masses the waveform needs.
_MASS_COLUMNS = (
    ("detector_frame_mass_1", "detector_frame_mass_2"),
    ("source_frame_mass_1", "source_frame_mass_2"),
)


def read_event_pool(
    path: str | Path, *, hdf5_dataset: str = "data"
) -> dict[str, NDArray[np.float64]]:
    """Read a population catalogue as an event pool for the simulator.

    The file is an HDF5 or CSV catalogue in the canonical gwmock-pop column
    names. Besides the waveform parameters it must hold ``distance`` (the
    luminosity distance in Mpc, renamed ``luminosity_distance`` here), the sky
    columns ``right_ascension``, ``declination`` and ``polarization_angle``,
    and either the detector-frame or the source-frame masses. Every other
    column is passed through unchanged. All columns are returned as float64.

    Parameters
    ----------
    path
        Catalogue path, ``.h5``, ``.hdf5`` or ``.csv``.
    hdf5_dataset
        Dataset name inside an HDF5 file.

    Returns
    -------
    dict of str to numpy.ndarray
        One-dimensional columns in this package's names.

    Raises
    ------
    ImportError
        If gwmock-pop is not installed.
    ValueError
        If a required column is missing.
    """
    try:
        from gwmock_pop import read_population_catalogue
    except ImportError as error:
        raise ImportError(
            "reading an event pool needs gwmock-pop; "
            "install it with the 'simulation' extra"
        ) from error

    catalogue = read_population_catalogue(path, hdf5_dataset=hdf5_dataset)
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
