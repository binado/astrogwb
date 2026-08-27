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
        minimum_frequency, maximum_frequency, reference_frequency, sampling_frequency, df,
        plus any scalar ``extra_attrs`` the producer stamped on (provenance)

``sample`` deliberately has no coordinate.

Files are written with ``to_netcdf(engine="h5netcdf", invalid_netcdf=True)``.
The data itself no longer needs ``invalid_netcdf`` -- there is no complex
compound type, so the format is now plain-netCDF compatible -- but the flag
is kept regardless; dropping it is a separate, unrelated change.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

__all__ = [
    "DOMAIN_FREQUENCY",
    "FORMAT_NAME",
    "RESERVED_ATTRS",
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

#: Slack, in ULPs of the largest frequency, allowed when checking that the grid
#: really is spaced by the stored ``df``. A grid built as ``arange(n) * df``
#: carries round-off of order ``eps * max|f|`` in each *stored value*, so
#: consecutive differences stray from ``df`` by about that much in absolute
#: terms. The error scales with the magnitude of the frequencies, not with
#: ``df``: a relative-to-``df`` tolerance would be tight by a factor of roughly
#: ``max|f| / df`` (the bin count) and would reject legitimate fine grids over
#: wide bands.
GRID_SPACING_TOLERANCE_ULP = 64.0

#: Attribute names ``make_catalog`` owns; ``extra_attrs`` may not shadow them.
RESERVED_ATTRS = frozenset(
    {
        "format_name",
        "domain",
        "approximant",
        "minimum_frequency",
        "maximum_frequency",
        "reference_frequency",
        "sampling_frequency",
        "df",
    }
)


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
    df: float,
    extra_attrs: Mapping[str, str | float | int] | None = None,
) -> xr.Dataset:
    """Build a waveform catalog Dataset from plain arrays.

    ``polarization_power`` has shape ``(nfreq, nsamples)`` -- frequency axis
    first, matching the on-disk layout. Not cast to float64 up front: a
    complex array is passed through as-is so ``validate_catalog`` can reject
    it below, instead of silently discarding the imaginary part.

    ``extra_attrs`` stamps producer-defined provenance onto the file. netCDF
    attributes are flat, so values must be str/int/float scalars -- encode
    anything structured (JSON, for instance) into a single string. Names in
    :data:`RESERVED_ATTRS` are rejected rather than overwritten: ``format_name``
    and ``domain`` are what makes a file readable at all.
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
            "df": float(df),
            **_validated_extra_attrs(extra_attrs),
        },
    )
    validate_catalog(catalog, label="waveform_catalog")
    return catalog


def _validated_extra_attrs(
    extra_attrs: Mapping[str, str | float | int] | None,
) -> dict[str, str | float | int]:
    """Reject reserved names and non-scalar values before they reach ``attrs``."""
    if not extra_attrs:
        return {}
    collisions = sorted(RESERVED_ATTRS.intersection(extra_attrs))
    if collisions:
        raise ValueError(
            "extra_attrs may not override reserved catalog attribute(s): "
            + ", ".join(collisions)
        )
    validated: dict[str, str | float | int] = {}
    for name, value in extra_attrs.items():
        # bool is an int subclass but netCDF has no boolean attribute type.
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            raise TypeError(
                f"extra_attrs[{name!r}] must be a str, int, or float scalar, got "
                f"{type(value).__name__}"
            )
        validated[name] = value
    return validated


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
    """Validate dimensions, shapes, and the uniform frequency grid."""
    if "frequency" not in catalog.coords:
        raise ValueError(f"{label}: missing 'frequency' coordinate")
    frequencies = catalog.coords["frequency"].values
    if frequencies.ndim != 1:
        raise ValueError(f"{label}: frequency coordinate must be one-dimensional")
    if frequencies.size == 0:
        raise ValueError(f"{label}: frequency coordinate must contain at least one bin")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError(f"{label}: frequencies must be finite")

    if "df" not in catalog.attrs:
        raise ValueError(
            f"{label}: missing required 'df' attribute; regenerate this catalog"
        )
    raw_df = catalog.attrs["df"]
    try:
        df = float(raw_df)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: df must be a finite positive scalar") from None
    if not np.isfinite(df) or df <= 0.0:
        raise ValueError(f"{label}: df must be a finite positive scalar")

    if frequencies.size > 1:
        differences = np.diff(frequencies)
        # Monotonicity is checked on its own rather than being left to the
        # tolerance below: for a df smaller than the tolerance, duplicate or
        # decreasing bins would satisfy |diff - df| <= tolerance and only fail
        # much later, in whatever consumes the grid.
        if not np.all(differences > 0.0):
            raise ValueError(f"{label}: frequencies must be strictly increasing")
        tolerance = (
            GRID_SPACING_TOLERANCE_ULP
            * np.finfo(np.float64).eps
            * max(1.0, float(np.max(np.abs(frequencies))))
        )
        if not np.all(np.abs(differences - df) <= tolerance):
            raise ValueError(
                f"{label}: frequencies must be uniformly spaced by df={df} Hz"
            )

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
