"""The h5py layer both catalog formats are written and read through.

Two artifacts live in this package -- a polarization-power catalog and a
spectral-density catalog -- and they are the same *kind* of file: root
attributes describing the waveform backend and the population that produced
them, plus a handful of flat datasets. Everything about that shape which does
not depend on which artifact it is lives here, so the two readers differ only
where the formats genuinely differ.

The scalar and JSON attribute codec is one layer down, in
:mod:`astrogwb._attrs`, which imports no h5py. This module is the only place
that opens a file.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from astrogwb._attrs import require_attrs, scalar_attr
from astrogwb.waveform import PolarizationPowerGenerator

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
    "WAVEFORM_ATTRS",
    "decoded_attrs",
    "h5py",
    "require_datasets",
    "waveform_attrs",
    "waveform_from_attrs",
    "write_h5",
]

#: The waveform descriptor, as attribute names. Both formats stamp all six;
#: the polarization-power catalog additionally tolerates an older file that
#: predates ``frequency_resolution`` (see ``waveform_from_attrs``).
WAVEFORM_ATTRS = (
    "approximant",
    "minimum_frequency",
    "maximum_frequency",
    "reference_frequency",
    "sampling_frequency",
    "frequency_resolution",
)


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


def waveform_attrs(
    generator: PolarizationPowerGenerator,
) -> dict[str, str | int | float]:
    """Encode a waveform descriptor as the six attributes both formats stamp."""
    return {
        "approximant": generator.approximant,
        "minimum_frequency": generator.minimum_frequency,
        "maximum_frequency": generator.maximum_frequency,
        "reference_frequency": generator.reference_frequency,
        "sampling_frequency": generator.sampling_frequency,
        "frequency_resolution": generator.frequency_resolution,
    }


def waveform_from_attrs(
    attrs: Mapping[str, Any],
    *,
    label: str,
    frequency_resolution_fallback: str | None = None,
) -> PolarizationPowerGenerator:
    """Rebuild the metadata-only waveform descriptor from file attributes.

    ``frequency_resolution_fallback`` names an attribute to read when
    ``frequency_resolution`` is absent, which is how the polarization-power
    catalog reads a file written before it stamped one. It is a default for an
    older file's *request*, never a cross-check: what the backend actually
    produced is the frequency dataset, and only that.
    """
    required = [name for name in WAVEFORM_ATTRS if name != "frequency_resolution"]
    required.append(frequency_resolution_fallback or "frequency_resolution")
    require_attrs(attrs, required, label=label, kind="waveform metadata")
    if "frequency_resolution" in attrs:
        frequency_resolution = attrs["frequency_resolution"]
    else:
        frequency_resolution = attrs[str(frequency_resolution_fallback)]
    try:
        return PolarizationPowerGenerator(
            approximant=str(scalar_attr(attrs["approximant"], name="approximant")),
            minimum_frequency=float(
                scalar_attr(attrs["minimum_frequency"], name="minimum_frequency")
            ),
            maximum_frequency=float(
                scalar_attr(attrs["maximum_frequency"], name="maximum_frequency")
            ),
            reference_frequency=float(
                scalar_attr(attrs["reference_frequency"], name="reference_frequency")
            ),
            sampling_frequency=float(
                scalar_attr(attrs["sampling_frequency"], name="sampling_frequency")
            ),
            frequency_resolution=float(
                scalar_attr(frequency_resolution, name="frequency_resolution")
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid waveform metadata: {error}") from error
