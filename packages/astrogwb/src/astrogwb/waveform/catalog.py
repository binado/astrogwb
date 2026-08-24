"""IO for the ``waveform_catalog`` xarray/HDF5 format.

Stores catalogs of frequency-domain waveform polarizations: complex ``plus`` /
``cross`` stacked over a ``polarization`` dimension, per-sample source
parameters stacked over a ``parameter`` dimension, and the waveform-generation
attributes. Pure IO: no derived quantities are computed here.

Catalogs are plain ``xr.Dataset`` objects (``WaveformCatalog`` is a type alias
for documentation value only) with the layout::

    Dimensions:           (polarization: 2, sample: N, frequency: F, parameter: P)
    Coordinates:
      * polarization      (polarization) <U5      'plus' 'cross'
      * frequency         (frequency)    float64  strictly increasing, Hz
      * parameter         (parameter)    <U..     'redshift' 'luminosity_distance' ...
    Data variables:
        polarizations     (polarization, sample, frequency) complex128
        source_parameters (sample, parameter)               float64
    Attributes:
        format_name, format_version, domain, approximant,
        minimum_frequency, maximum_frequency, reference_frequency, sampling_frequency

``sample`` deliberately has no coordinate.

Files are written with ``to_netcdf(engine="h5netcdf", invalid_netcdf=True)``,
which stores complex128 as an HDF5 compound type with members named ``r`` and
``i`` -- the same on-disk representation the retired ``pluscross`` package used.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

__all__ = [
    "DOMAIN_FREQUENCY",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "WaveformCatalog",
    "load_catalog",
    "make_catalog",
    "open_catalog",
    "save_catalog",
    "validate_catalog",
]

#: Type alias for documentation value only -- catalogs are plain Datasets.
WaveformCatalog = xr.Dataset

FORMAT_NAME = "waveform_catalog"
FORMAT_VERSION = 2
DOMAIN_FREQUENCY = "frequency"

_POLARIZATIONS = ("plus", "cross")


def make_catalog(
    *,
    frequencies: NDArray[np.float64],
    plus: NDArray[np.complex128],
    cross: NDArray[np.complex128],
    source_parameters: dict[str, NDArray[np.float64]],
    approximant: str,
    minimum_frequency: float,
    maximum_frequency: float,
    reference_frequency: float,
    sampling_frequency: float,
) -> xr.Dataset:
    """Build a waveform catalog Dataset from plain arrays.

    ``plus`` and ``cross`` have shape ``(nsamples, nfreq)`` -- sample axis
    first, matching the on-disk C-order layout of ``polarizations``.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    plus = np.asarray(plus, dtype=np.complex128)
    cross = np.asarray(cross, dtype=np.complex128)
    if plus.shape != cross.shape:
        raise ValueError(
            f"waveform_catalog: plus {plus.shape} and cross {cross.shape} shapes differ"
        )
    nsamples = plus.shape[0]
    names = list(source_parameters)
    if names:
        parameters = np.stack(
            [np.asarray(source_parameters[name], dtype=np.float64) for name in names],
            axis=1,
        )
    else:
        parameters = np.empty((nsamples, 0), dtype=np.float64)

    catalog = xr.Dataset(
        {
            "polarizations": (
                ("polarization", "sample", "frequency"),
                np.stack([plus, cross], axis=0),
            ),
            "source_parameters": (("sample", "parameter"), parameters),
        },
        coords={
            "polarization": list(_POLARIZATIONS),
            "frequency": frequencies,
            "parameter": names,
        },
        attrs={
            "format_name": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "domain": DOMAIN_FREQUENCY,
            "approximant": approximant,
            "minimum_frequency": float(minimum_frequency),
            "maximum_frequency": float(maximum_frequency),
            "reference_frequency": float(reference_frequency),
            "sampling_frequency": float(sampling_frequency),
        },
    )
    validate_catalog(catalog, label="waveform_catalog")
    return catalog


def save_catalog(
    path: str | Path,
    catalog: xr.Dataset,
    *,
    compression: str | None = None,
) -> None:
    """Write ``catalog`` to ``path`` in waveform_catalog format v2.

    Polarization data is uncompressed by default. Pass ``compression`` (for
    example ``"gzip"``) to opt into an HDF5 compression filter; HDF5 then
    picks the on-disk chunking automatically.
    """
    validate_catalog(catalog, label="waveform_catalog")
    encoding = (
        {"polarizations": {"compression": compression}}
        if compression is not None
        else None
    )

    with warnings.catch_warnings():
        # invalid_netcdf=True stores complex128 as an HDF5 compound {r, i}
        # type, which is exactly what we want -- not valid CF-netCDF, but a
        # deliberate, documented on-disk format.
        warnings.filterwarnings("ignore", category=UserWarning)
        catalog.to_netcdf(
            path, engine="h5netcdf", invalid_netcdf=True, encoding=encoding
        )


def _check_format(catalog: xr.Dataset, *, label: str) -> None:
    """Guard against opening a stale or foreign file. Must run before anything else."""
    format_name = catalog.attrs.get("format_name")
    if format_name != FORMAT_NAME:
        raise ValueError(
            f"{label}: format_name is {format_name!r}, expected {FORMAT_NAME!r}"
        )
    version = catalog.attrs.get("format_version")
    if version is None or int(version) != FORMAT_VERSION:
        raise ValueError(
            f"{label}: format_version is {version!r}, expected {FORMAT_VERSION} "
            "-- v1 (pluscross) catalogs must be regenerated"
        )
    domain = catalog.attrs.get("domain")
    if domain != DOMAIN_FREQUENCY:
        raise ValueError(
            f"{label}: domain is {domain!r}, expected {DOMAIN_FREQUENCY!r}"
        )


def load_catalog(path: str | Path) -> xr.Dataset:
    """Read a waveform_catalog v2 file eagerly into memory."""
    label = Path(path).name
    catalog = xr.load_dataset(path, engine="h5netcdf")
    _check_format(catalog, label=label)
    validate_catalog(catalog, label=label)
    return catalog


def open_catalog(path: str | Path) -> xr.Dataset:
    """Open a waveform_catalog v2 file lazily; polarizations are read on demand."""
    label = Path(path).name
    catalog = xr.open_dataset(path, engine="h5netcdf")
    _check_format(catalog, label=label)
    validate_catalog(catalog, label=label)
    return catalog


def validate_catalog(catalog: xr.Dataset, *, label: str) -> None:
    """Validate dimension names, shapes, and frequency monotonicity."""
    if "frequency" not in catalog.coords:
        raise ValueError(f"{label}: missing 'frequency' coordinate")
    frequencies = catalog.coords["frequency"].values
    if frequencies.ndim != 1:
        raise ValueError(f"{label}: frequency coordinate must be one-dimensional")
    if frequencies.shape[0] > 1 and not np.all(np.diff(frequencies) > 0.0):
        raise ValueError(f"{label}: frequencies must be strictly increasing")

    if "polarizations" not in catalog:
        raise ValueError(f"{label}: missing 'polarizations' data variable")
    polarizations = catalog["polarizations"]
    if set(polarizations.dims) != {"polarization", "sample", "frequency"}:
        raise ValueError(
            f"{label}: 'polarizations' must have dims "
            f"(polarization, sample, frequency), got {polarizations.dims}"
        )
    if catalog.sizes["polarization"] != 2:
        raise ValueError(
            f"{label}: 'polarization' dim must have size 2, got "
            f"{catalog.sizes['polarization']}"
        )
    if catalog.sizes["frequency"] != frequencies.shape[0]:
        raise ValueError(
            f"{label}: 'polarizations' frequency axis "
            f"({catalog.sizes['frequency']}) does not match the frequency "
            f"coordinate ({frequencies.shape[0]})"
        )

    if "source_parameters" in catalog:
        source_parameters = catalog["source_parameters"]
        if set(source_parameters.dims) != {"sample", "parameter"}:
            raise ValueError(
                f"{label}: 'source_parameters' must have dims "
                f"(sample, parameter), got {source_parameters.dims}"
            )
        if source_parameters.sizes["sample"] != polarizations.sizes["sample"]:
            raise ValueError(
                f"{label}: 'source_parameters' sample axis "
                f"({source_parameters.sizes['sample']}) does not match "
                f"'polarizations' sample axis ({polarizations.sizes['sample']})"
            )
