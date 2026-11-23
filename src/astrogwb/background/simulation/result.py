"""The result of a background simulation and its HDF5 file format."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from astrogwb import __version__
from astrogwb.background.simulation.config import SimulationConfig
from astrogwb.utils.io import atomic_h5, h5py

__all__ = ["BackgroundDraw"]

#: The layout version of the files :meth:`BackgroundDraw.write` writes.
_FORMAT_VERSION = 2

#: Root attributes :meth:`BackgroundDraw.write` owns.
_RESERVED_ATTRS = frozenset({"metadata", "version", "format_version"})


class BackgroundDraw(NamedTuple):
    """One realized background projected onto a detector network.

    Attributes
    ----------
    frequencies
        The frequency grid in Hz, ``(F,)``.
    pair_spectral_density
        Pair spectra :math:`Y_{ab}` in :math:`\\mathrm{Hz}^{-1}`, with shape
        ``(pairs, frequencies)``. The pair axis follows
        :func:`~astrogwb.detector.network_pairs`.
    segment_counts
        The number of events in each segment, ``(num_segments,)``. The
        projection of :func:`project_background` has one entry for the whole
        observation.
    """

    frequencies: NDArray[np.float64]
    pair_spectral_density: NDArray[np.float64]
    segment_counts: NDArray[np.int64]

    def write(
        self,
        path: str | Path,
        config: SimulationConfig,
        /,
        **attrs: str | float,
    ) -> Path:
        """Write the background and the configuration it was simulated with, atomically.

        The file holds the datasets ``frequencies``, ``segment_counts`` and
        ``pair_spectral_density``. Its root attributes
        are ``metadata``, the configuration as JSON, ``format_version``, the
        integer layout version :meth:`read` keys on, and ``version``, the
        astrogwb version that wrote it (informational), plus any ``attrs``.

        Parameters
        ----------
        path
            File path.
        config
            The configuration the background was simulated with.
        **attrs
            Extra root attributes, written in the same atomic write and read
            back by :meth:`read_attrs`.

        Returns
        -------
        pathlib.Path
            The file written.

        Raises
        ------
        ValueError
            If ``attrs`` uses a reserved key (``metadata``, ``version`` or
            ``format_version``). Nothing is written.
        """
        clashes = sorted(_RESERVED_ATTRS.intersection(attrs))
        if clashes:
            raise ValueError(f"attrs may not use the reserved keys {clashes}")
        with atomic_h5(path) as file:
            file.attrs["metadata"] = json.dumps(config.to_metadata())
            file.attrs["version"] = __version__
            file.attrs["format_version"] = _FORMAT_VERSION
            for name, value in attrs.items():
                file.attrs[name] = value
            file.create_dataset("frequencies", data=self.frequencies)
            file.create_dataset("segment_counts", data=self.segment_counts)
            file.create_dataset(
                "pair_spectral_density", data=self.pair_spectral_density
            )
        return Path(path)

    @classmethod
    def read(cls, path: str | Path) -> tuple[BackgroundDraw, SimulationConfig]:
        """Read a background and its configuration written by :meth:`write`.

        Parameters
        ----------
        path
            File path.

        Returns
        -------
        BackgroundDraw
            The background's pair spectra in canonical order.
        SimulationConfig
            The configuration it was simulated with.

        Raises
        ------
        ValueError
            If the file's ``format_version`` is missing or unsupported.
        """
        with h5py.File(path, "r") as file:
            _check_format(path, file.attrs)
            config = SimulationConfig.from_metadata(json.loads(file.attrs["metadata"]))
            draw = cls(
                np.asarray(file["frequencies"][()], dtype=np.float64),
                np.asarray(file["pair_spectral_density"][()], dtype=np.float64),
                np.asarray(file["segment_counts"][()], dtype=np.int64),
            )
        return draw, config

    @staticmethod
    def read_attrs(path: str | Path) -> dict[str, str | float | int]:
        """The extra root attributes :meth:`write` stored as ``attrs``.

        Parameters
        ----------
        path
            File path.

        Returns
        -------
        dict
            The attributes other than the reserved ``metadata``, ``version``
            and ``format_version``, as Python scalars.

        Raises
        ------
        ValueError
            As for :meth:`read`, if the file's format is not readable.
        """
        with h5py.File(path, "r") as file:
            _check_format(path, file.attrs)
            return {
                name: value.item() if isinstance(value, np.generic) else value
                for name, value in file.attrs.items()
                if name not in _RESERVED_ATTRS
            }


def _check_format(path: str | Path, attrs: Mapping[str, object]) -> None:
    """Raise unless the file's layout is one this version can read."""
    version = attrs.get("version")
    format_version = attrs.get("format_version")
    if format_version == _FORMAT_VERSION:
        return
    raise ValueError(
        f"{path} has format_version {format_version} (written by astrogwb "
        f"{version}); astrogwb {__version__} reads format version {_FORMAT_VERSION}; "
        "regenerate the background"
    )
