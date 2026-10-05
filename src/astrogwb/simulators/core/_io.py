"""Direct HDF5 persistence for the two catalog formats.

Both writers store their complete Pydantic record in a root ``metadata``
JSON attribute. Dataset layout and column ordering remain format-specific.

Structural validation is deliberately split. Cross-field invariants -- axis
agreement, draw counts, the frequency grid -- belong to each record's
``__post_init__``, so they hold for an object built in memory and not only for
one that has been through a file. The ``validate_*_file`` functions check what
only a file can get wrong: a missing dataset or attribute, a foreign format
name, a payload serialized at the wrong dtype.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

from astrogwb._attrs import (
    DOMAIN_FREQUENCY,
    FORMAT_NAME_ATTR,
    json_array_attr,
    require_attrs,
    require_format,
    stack_columns,
    unstack_columns,
)
from astrogwb.simulators.core._hdf5 import (
    h5py,
    require_datasets,
    write_h5,
)
from astrogwb.simulators.polarization_power.catalog import (
    REDSHIFT_SITE,
    PolarizationPowerCatalog,
)
from astrogwb.simulators.polarization_power.metadata import CatalogMetadata
from astrogwb.simulators.spectra.catalog import (
    SpectralDensityCatalog,
)
from astrogwb.simulators.spectra.metadata import SpectraMetadata

__all__ = [
    "CATALOG_FORMAT_NAME",
    "SPECTRAL_DENSITY_FORMAT_NAME",
    "load_polarization_power_catalog",
    "load_spectral_density_catalog",
    "save_polarization_power_catalog",
    "save_spectral_density_catalog",
    "validate_catalog_file",
    "validate_spectral_density_file",
]

#: The name list every format persists alongside its stacked parameter matrix.
PARAMETER_NAMES_ATTR = "source_parameter_names"

# --------------------------------------------------------------------- #
# Polarization-power catalogs
# --------------------------------------------------------------------- #
CATALOG_FORMAT_NAME = "astrogwb_catalog_v9"
CATALOG_DATASETS = ("frequency", "polarization_power", "source_parameters")

METADATA_ATTR = "metadata"


def save_polarization_power_catalog(
    catalog: PolarizationPowerCatalog,
    path: str | Path,
    *,
    compression: str | None = None,
) -> None:
    """Write one catalog in the v9 direct-HDF5 format."""
    names, source_parameters = stack_columns(
        catalog.source_parameters, rows=catalog.num_samples
    )
    metadata = catalog.metadata
    attrs: dict[str, str | int | float] = {
        FORMAT_NAME_ATTR: CATALOG_FORMAT_NAME,
        "domain": DOMAIN_FREQUENCY,
        METADATA_ATTR: metadata.model_dump_json(),
        PARAMETER_NAMES_ATTR: json.dumps(names),
    }
    write_h5(
        path,
        attrs=attrs,
        datasets={
            "frequency": catalog.frequencies,
            "polarization_power": catalog.polarization_power,
            "source_parameters": source_parameters,
        },
        compression=compression,
    )


def load_polarization_power_catalog[C: PolarizationPowerCatalog](
    cls: type[C], path: str | Path
) -> C:
    """Read and structurally validate one catalog, rebuilding its model record."""
    label = Path(path).name
    with h5py.File(path, "r") as handle:
        validate_catalog_file(handle, label=label)
        attrs = handle.attrs
        names = json_array_attr(
            attrs[PARAMETER_NAMES_ATTR], label=label, name=PARAMETER_NAMES_ATTR
        )
        metadata = _read_metadata(handle, CatalogMetadata, label=label)
        power = np.asarray(handle["polarization_power"])
        catalog = cls(
            source_parameters=unstack_columns(
                np.asarray(handle["source_parameters"]), names
            ),
            polarization_power=power,
            frequencies=np.asarray(handle["frequency"]),
            _metadata=metadata,
        )
    metadata.population.check_registered()
    return catalog


def validate_catalog_file(handle: h5py.File | h5py.Group, *, label: str) -> None:
    """Validate HDF5 layout, metadata, and serialized dtypes."""
    require_format(
        handle.attrs,
        label=label,
        format_name=CATALOG_FORMAT_NAME,
        domain=DOMAIN_FREQUENCY,
    )
    require_datasets(handle, CATALOG_DATASETS, label=label)
    frequency = handle["frequency"]
    if frequency.ndim != 1:
        raise ValueError(f"{label}: frequency dataset must be one-dimensional")
    metadata = _read_metadata(handle, CatalogMetadata, label=label)
    power = handle["polarization_power"]
    if power.ndim != 2 or power.shape[0] != frequency.shape[0]:
        raise ValueError(
            f"{label}: 'polarization_power' must have shape (frequency, sample)"
        )
    if not np.issubdtype(power.dtype, np.number) or np.issubdtype(
        power.dtype, np.complexfloating
    ):
        raise ValueError(f"{label}: 'polarization_power' must be real-valued")
    source = handle["source_parameters"]
    if source.ndim != 2 or source.shape[0] != power.shape[1]:
        raise ValueError(
            f"{label}: 'source_parameters' must have shape (sample, parameter)"
        )
    if source.dtype != np.dtype(np.float64):
        raise ValueError(f"{label}: 'source_parameters' must be serialized as float64")

    if metadata.num_samples != source.shape[0]:
        raise ValueError(
            f"{label}: metadata num_samples ({metadata.num_samples}) does not match "
            f"the sample dimension ({source.shape[0]})"
        )
    names = _column_names(handle.attrs, label=label, columns=source.shape[1])
    if REDSHIFT_SITE not in names:
        raise ValueError(f"{label}: missing the {REDSHIFT_SITE!r} source parameter")


# --------------------------------------------------------------------- #
# Spectral-density catalogs
# --------------------------------------------------------------------- #
SPECTRAL_DENSITY_FORMAT_NAME = "astrogwb_spectral_density_v5"
SPECTRAL_DENSITY_DATASETS = (
    "frequency",
    "spectral_density",
    "n_events",
    "total_merger_rate",
    "hyperparameters",
)


def save_spectral_density_catalog(
    catalog: SpectralDensityCatalog,
    path: str | Path,
    *,
    compression: str | None = None,
) -> None:
    """Write one spectral-density catalog to HDF5."""
    names, hyperparameters = stack_columns(
        catalog.hyperparameters, rows=catalog.num_draws
    )
    metadata = catalog.metadata
    attrs: dict[str, str | int | float] = {
        FORMAT_NAME_ATTR: SPECTRAL_DENSITY_FORMAT_NAME,
        "domain": DOMAIN_FREQUENCY,
        METADATA_ATTR: metadata.model_dump_json(),
        PARAMETER_NAMES_ATTR: json.dumps(names),
    }
    write_h5(
        path,
        attrs=attrs,
        datasets={
            "frequency": catalog.frequencies,
            "spectral_density": catalog.spectral_density,
            "n_events": catalog.n_events,
            "total_merger_rate": catalog.total_merger_rate,
            "hyperparameters": hyperparameters,
        },
        compression=compression,
    )


def load_spectral_density_catalog[C: SpectralDensityCatalog](
    cls: type[C], path: str | Path
) -> C:
    """Read and validate one spectral-density catalog, rebuilding its record."""
    label = Path(path).name
    with h5py.File(path, "r") as handle:
        validate_spectral_density_file(handle, label=label)
        attrs = handle.attrs
        names = json_array_attr(
            attrs[PARAMETER_NAMES_ATTR], label=label, name=PARAMETER_NAMES_ATTR
        )
        metadata = _read_metadata(handle, SpectraMetadata, label=label)
        spectral_density = np.asarray(handle["spectral_density"])
        catalog = cls(
            spectral_density=spectral_density,
            frequencies=np.asarray(handle["frequency"]),
            n_events=np.asarray(handle["n_events"]),
            total_merger_rate=np.asarray(handle["total_merger_rate"]),
            hyperparameters=unstack_columns(
                np.asarray(handle["hyperparameters"]), names
            ),
            _metadata=metadata,
        )
    metadata.population.check_registered()
    return catalog


def validate_spectral_density_file(
    handle: h5py.File | h5py.Group, *, label: str
) -> None:
    """Validate HDF5 layout, metadata, and serialized dtypes."""
    require_format(
        handle.attrs,
        label=label,
        format_name=SPECTRAL_DENSITY_FORMAT_NAME,
        domain=DOMAIN_FREQUENCY,
    )
    require_datasets(handle, SPECTRAL_DENSITY_DATASETS, label=label)
    _read_metadata(handle, SpectraMetadata, label=label)

    hyperparameters = handle["hyperparameters"]
    if hyperparameters.ndim != 2:
        raise ValueError(
            f"{label}: 'hyperparameters' must have shape (draw, parameter)"
        )
    if hyperparameters.dtype != np.dtype(np.float64):
        raise ValueError(f"{label}: 'hyperparameters' must be serialized as float64")
    _column_names(handle.attrs, label=label, columns=hyperparameters.shape[1])


# --------------------------------------------------------------------- #
# Shared between the two readers
# --------------------------------------------------------------------- #
def _read_metadata[M: BaseModel](
    handle: h5py.File | h5py.Group, model: type[M], *, label: str
) -> M:
    """Validate the complete JSON record, identifying the file on failure."""
    require_attrs(handle.attrs, (METADATA_ATTR,), label=label, kind="catalog metadata")
    try:
        return model.model_validate_json(handle.attrs[METADATA_ATTR])
    except (ValidationError, TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid catalog metadata: {error}") from error


def _column_names(attrs: Any, *, label: str, columns: int) -> list[str]:
    """Read the column-name list and check it describes the stacked matrix."""
    names = json_array_attr(
        attrs.get(PARAMETER_NAMES_ATTR, ""), label=label, name=PARAMETER_NAMES_ATTR
    )
    if len(names) != columns or len(set(names)) != len(names):
        raise ValueError(
            f"{label}: parameter ordering does not match the stored parameter matrix"
        )
    return [str(name) for name in names]
