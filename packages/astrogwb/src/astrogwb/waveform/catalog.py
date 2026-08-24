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
        snr                (sample, detector) float64   # optional
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

import math
import warnings
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Any

import h5netcdf
import h5py
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

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
_SAMPLE_WRITE_TARGET_BYTES = 64 * 1024**2


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
    sample_order: ArrayLike | None = None,
) -> None:
    """Write ``catalog`` to ``path`` in waveform_catalog format.

    Polarization power is uncompressed by default. Pass ``compression`` (for
    example ``"gzip"``) to opt into an HDF5 compression filter. When
    ``sample_order`` is supplied, it must be a complete permutation of the
    sample axis; sample-dependent variables are written in bounded blocks so
    lazy waveform power is never fully materialized.
    """
    validate_catalog(catalog, label="waveform_catalog")
    if sample_order is not None:
        order = _validate_sample_order(sample_order, catalog.sizes["sample"])
        _save_catalog_in_sample_order(
            Path(path), catalog, order=order, compression=compression
        )
        return

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


def _validate_sample_order(order: ArrayLike, n_samples: int) -> NDArray[np.intp]:
    array = np.asarray(order)
    if array.ndim != 1 or array.shape[0] != n_samples:
        raise ValueError(
            "sample_order must be one-dimensional with one entry per sample"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("sample_order must contain integer indices")
    indices = array.astype(np.intp, copy=False)
    if not np.array_equal(np.sort(indices), np.arange(n_samples, dtype=np.intp)):
        raise ValueError("sample_order must be a complete permutation")
    return indices


def _save_catalog_in_sample_order(
    path: Path,
    catalog: xr.Dataset,
    *,
    order: NDArray[np.intp],
    compression: str | None,
) -> None:
    """Write a sample permutation without loading any full sample-sized variable."""
    with h5netcdf.File(path, "w", invalid_netcdf=True) as output:
        output.dimensions = dict(catalog.sizes)
        for name, value in catalog.attrs.items():
            output.attrs[name] = value

        for name, variable in catalog.variables.items():
            variable_name = str(name)
            dtype = _storage_dtype(variable.dtype)
            options = _storage_options(
                variable, name=variable_name, compression=compression
            )
            fillvalue = variable.encoding.get("_FillValue")
            destination = output.create_variable(
                variable_name,
                variable.dims,
                dtype=dtype,
                fillvalue=fillvalue,
                **options,
            )
            for attribute, value in variable.attrs.items():
                destination.attrs[attribute] = value

            if "sample" not in variable.dims:
                _write_variable(destination, Ellipsis, variable.values)
                continue

            sample_axis = variable.dims.index("sample")
            block_size = _sample_block_size(variable, catalog.sizes)
            for start in range(0, order.size, block_size):
                stop = min(start + block_size, order.size)
                values = variable.isel(sample=order[start:stop]).values
                target = [slice(None)] * variable.ndim
                target[sample_axis] = slice(start, stop)
                _write_variable(destination, tuple(target), values)


def _storage_dtype(dtype: np.dtype) -> np.dtype | str:
    if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_):
        return h5py.string_dtype(encoding="utf-8")
    return dtype


def _storage_options(
    variable: xr.Variable, *, name: str, compression: str | None
) -> dict[str, object]:
    encoding = variable.encoding
    options: dict[str, object] = {}
    chunks = encoding.get("chunksizes")
    if chunks is not None:
        options["chunks"] = tuple(
            min(int(chunk), int(size))
            for chunk, size in zip(chunks, variable.shape, strict=True)
        )
    for key in ("compression", "compression_opts", "shuffle", "fletcher32"):
        value = encoding.get(key)
        if value not in (None, False):
            options[key] = value
    if encoding.get("zlib"):
        options["compression"] = "gzip"
        options["compression_opts"] = int(encoding.get("complevel", 4))
    if name == "polarization_power" and compression is not None:
        options["compression"] = compression
    return options


def _sample_block_size(variable: xr.Variable, sizes: Mapping[Hashable, int]) -> int:
    other_elements = math.prod(
        sizes[dimension] for dimension in variable.dims if dimension != "sample"
    )
    itemsize = max(1, variable.dtype.itemsize)
    bytes_per_sample = max(1, other_elements * itemsize)
    return max(1, _SAMPLE_WRITE_TARGET_BYTES // bytes_per_sample)


def _write_variable(destination: h5netcdf.Variable, key: Any, values: object) -> None:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.str_):
        array = array.astype(object)
    destination[key] = array


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

    if "snr" in catalog:
        if "detector" not in catalog.coords:
            raise ValueError(f"{label}: 'snr' requires a 'detector' coordinate")
        snr = catalog["snr"]
        if snr.dims != ("sample", "detector"):
            raise ValueError(
                f"{label}: 'snr' must have dims (sample, detector), got {snr.dims}"
            )
        if not np.issubdtype(snr.dtype, np.floating):
            raise ValueError(f"{label}: 'snr' must be floating-point")
        if snr.sizes["sample"] != polarization_power.sizes["sample"]:
            raise ValueError(
                f"{label}: 'snr' sample axis ({snr.sizes['sample']}) does not "
                f"match 'polarization_power' ({polarization_power.sizes['sample']})"
            )
        detector = np.asarray(catalog.coords["detector"].values)
        if detector.ndim != 1 or detector.shape[0] != snr.sizes["detector"]:
            raise ValueError(f"{label}: 'detector' must be one-dimensional")
        detector_names = [str(name) for name in detector]
        if any(not name for name in detector_names) or len(set(detector_names)) != len(
            detector_names
        ):
            raise ValueError(f"{label}: detector names must be non-empty and unique")
        snr_values = np.asarray(snr.values)
        if not np.all(np.isfinite(snr_values)) or np.any(snr_values < 0.0):
            raise ValueError(f"{label}: 'snr' values must be finite and non-negative")
