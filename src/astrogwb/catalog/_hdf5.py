"""The h5py layer both catalog formats are written and read through.

Two artifacts live in this package -- a polarization-power catalog and a
spectral-density catalog -- and they are the same *kind* of file: root
attributes describing the waveform backend and the population that produced
them, plus a handful of flat datasets. What is here is the h5py mechanics that
shape implies, so the two readers differ only where the formats genuinely
differ.

The metadata codecs themselves are not here. Each record encodes itself --
:meth:`astrogwb.waveform.PolarizationPowerGenerator.to_attrs` and
:meth:`astrogwb.metadata.PopulationMetadata.to_attrs` -- over the scalar and
JSON primitives in :mod:`astrogwb._attrs`, which imports no h5py. That is what
keeps the waveform and population layers free of a serialization dependency,
and leaves this module as the one place that opens a file.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from astrogwb._attrs import scalar_attr

try:
    import h5py
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "astrogwb catalog file I/O needs h5py. Install it with the 'io' extra: "
        "pip install 'astrogwb[io]'"
    ) from error

#: ``h5py`` is re-exported so this module is the one place that resolves it,
#: guard message included: a reader importing it from here cannot bypass the
#: try above.
__all__ = [
    "decoded_attrs",
    "h5py",
    "require_datasets",
    "write_h5",
]


def write_h5(
    path: str | Path,
    *,
    attrs: Mapping[str, str | int | float],
    datasets: Mapping[str, ArrayLike],
    compression: str | None = None,
) -> None:
    """Write one artifact: root attributes, then datasets, replacing ``path``."""
    with h5py.File(path, "w") as handle:
        for name, value in attrs.items():
            handle.attrs[name] = value
        for name, values in datasets.items():
            handle.create_dataset(
                name, data=np.asarray(values), compression=compression
            )


def decoded_attrs(handle: h5py.Group) -> dict[str, str | int | float]:
    """Read every root attribute, reduced to plain Python scalars."""
    return {
        str(name): scalar_attr(value, name=str(name))
        for name, value in handle.attrs.items()
    }


def require_datasets(
    handle: h5py.File | h5py.Group, names: tuple[str, ...], *, label: str
) -> None:
    """Raise on the first dataset the file does not contain."""
    for name in names:
        if name not in handle:
            raise ValueError(f"{label}: missing {name!r} dataset")
