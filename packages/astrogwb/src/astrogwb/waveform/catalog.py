"""IO for the ``waveform_catalog`` xarray/HDF5 format.

Stores catalogs of frequency-domain polarization power ``|h+|^2 + |hx|^2``,
per-sample source parameters stacked over a ``parameter`` dimension, and the
waveform-generation attributes. Pure IO: no derived quantities are computed
here -- the power reduction happens at generation time, before the catalog
is built.

Catalogs are plain ``xr.Dataset`` objects (``WaveformCatalog`` is a type alias
for documentation value only) with the layout::

    Dimensions:            (frequency: F, sample: N, parameter: P)
    Coordinates:
      * frequency          (frequency) float64  strictly increasing, Hz
      * parameter          (parameter) <U..     'redshift' 'luminosity_distance' ...
    Data variables:
        polarization_power (frequency, sample) float64   # |h+|^2 + |hx|^2
        source_parameters  (sample, parameter) float64
    Attributes:
        format_name, domain, approximant,
        minimum_frequency, maximum_frequency, reference_frequency, sampling_frequency

``sample`` deliberately has no coordinate.

Files are written with ``to_netcdf(engine="h5netcdf", invalid_netcdf=True)``.
The data itself no longer needs ``invalid_netcdf`` -- there is no complex
compound type, so the format is now plain-netCDF compatible -- but the flag
is kept regardless; dropping it is a separate, unrelated change.
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
DOMAIN_FREQUENCY = "frequency"


def make_catalog(
    *,
    frequencies: NDArray[np.float64],
    polarization_power: NDArray[np.float64],
    source_parameters: dict[str, NDArray[np.float64]],
    approximant: str,
    minimum_frequency: float,
    maximum_frequency: float,
    reference_frequency: float,
    sampling_frequency: float,
) -> xr.Dataset:
    """Build a waveform catalog Dataset from plain arrays.

    ``polarization_power`` has shape ``(nfreq, nsamples)`` -- frequency axis
    first, matching the on-disk layout. Not cast to float64 up front: a
    complex array is passed through as-is so ``validate_catalog`` can reject
    it below, instead of silently discarding the imaginary part.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    polarization_power = np.asarray(polarization_power)
    nsamples = polarization_power.shape[1]
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
            "polarization_power": (
                ("frequency", "sample"),
                polarization_power,
            ),
            "source_parameters": (("sample", "parameter"), parameters),
        },
        coords={
            "frequency": frequencies,
            "parameter": names,
        },
        attrs={
            "format_name": FORMAT_NAME,
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
    """Write ``catalog`` to ``path`` in waveform_catalog format.

    Polarization power is uncompressed by default. Pass ``compression`` (for
    example ``"gzip"``) to opt into an HDF5 compression filter; HDF5 then
    picks the on-disk chunking automatically.
    """
    validate_catalog(catalog, label="waveform_catalog")
    encoding = (
        {"polarization_power": {"compression": compression}}
        if compression is not None
        else None
    )

    with warnings.catch_warnings():
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
    domain = catalog.attrs.get("domain")
    if domain != DOMAIN_FREQUENCY:
        raise ValueError(
            f"{label}: domain is {domain!r}, expected {DOMAIN_FREQUENCY!r}"
        )


def load_catalog(path: str | Path) -> xr.Dataset:
    """Read a waveform_catalog file eagerly into memory."""
    label = Path(path).name
    catalog = xr.load_dataset(path, engine="h5netcdf")
    _check_format(catalog, label=label)
    validate_catalog(catalog, label=label)
    return catalog


def open_catalog(path: str | Path) -> xr.Dataset:
    """Open a waveform_catalog file lazily; polarization power is read on demand."""
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

    if "polarization_power" not in catalog:
        raise ValueError(f"{label}: missing 'polarization_power' data variable")
    polarization_power = catalog["polarization_power"]
    if set(polarization_power.dims) != {"frequency", "sample"}:
        raise ValueError(
            f"{label}: 'polarization_power' must have dims "
            f"(frequency, sample), got {polarization_power.dims}"
        )
    if np.issubdtype(polarization_power.dtype, np.complexfloating):
        raise ValueError(
            f"{label}: 'polarization_power' must be real-valued, got "
            f"{polarization_power.dtype} -- pass |h+|^2 + |hx|^2, not raw "
            "complex polarizations"
        )
    if catalog.sizes["frequency"] != frequencies.shape[0]:
        raise ValueError(
            f"{label}: 'polarization_power' frequency axis "
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
        if source_parameters.sizes["sample"] != polarization_power.sizes["sample"]:
            raise ValueError(
                f"{label}: 'source_parameters' sample axis "
                f"({source_parameters.sizes['sample']}) does not match "
                f"'polarization_power' sample axis "
                f"({polarization_power.sizes['sample']})"
            )
