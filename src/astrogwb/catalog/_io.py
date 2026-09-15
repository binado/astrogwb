"""Direct HDF5 persistence for the two catalog formats.

Both writers stamp the same three metadata blocks -- format identity, the
waveform descriptor, and the population record -- and differ only in their
datasets and in the handful of attributes each artifact alone carries. What is
shared lives in :mod:`astrogwb.catalog._hdf5` and
:mod:`astrogwb.populations.record`; what is here is per-format: which datasets
exist, which attributes are required, the shape and dtype checks a reader
enforces before constructing a record.

Structural validation is deliberately split. Cross-field invariants -- axis
agreement, draw counts, the uniform grid -- belong to each record's
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

from astrogwb._attrs import (
    DOMAIN_FREQUENCY,
    FORMAT_NAME_ATTR,
    int_attr,
    json_array_attr,
    json_object_attr,
    require_attrs,
    require_format,
    stack_columns,
    unstack_columns,
)
from astrogwb.catalog._hdf5 import (
    decoded_attrs,
    h5py,
    require_datasets,
    waveform_attrs,
    waveform_from_attrs,
    write_h5,
)
from astrogwb.catalog.polarization_power import REDSHIFT_SITE, PolarizationPowerCatalog
from astrogwb.catalog.spectral_density import (
    SpectralDensityCatalog,
)
from astrogwb.populations.record import POPULATION_ATTRS, PopulationRecord

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
CATALOG_FORMAT_NAME = "astrogwb_catalog_v6"
CATALOG_DATASETS = ("frequency", "polarization_power", "source_parameters")

#: Written on every save. It equals the ``df`` attribute for every current
#: waveform descriptor (see :func:`waveform_from_attrs`).
FREQUENCY_RESOLUTION_ATTR = "frequency_resolution"
#: Measured from the ``frequency`` dataset on write and never read back into a
#: computation -- the dataset is the truth. It doubles as the fallback a
#: pre-``frequency_resolution`` file's waveform descriptor is rebuilt from.
DF_ATTR = "df"
POPULATION_NUM_SAMPLES_ATTR = "population_num_samples"
POPULATION_PARAMS_ATTR = "population_params"

REQUIRED_CATALOG_ATTRS = (
    *POPULATION_ATTRS,
    POPULATION_NUM_SAMPLES_ATTR,
    POPULATION_PARAMS_ATTR,
)


def save_polarization_power_catalog(
    catalog: PolarizationPowerCatalog,
    path: str | Path,
    *,
    compression: str | None = None,
) -> None:
    """Write one catalog in the v6 direct-HDF5 format."""
    names, source_parameters = stack_columns(
        catalog.source_parameters, rows=catalog.num_samples
    )
    attrs: dict[str, str | int | float] = {
        FORMAT_NAME_ATTR: CATALOG_FORMAT_NAME,
        "domain": DOMAIN_FREQUENCY,
        **waveform_attrs(catalog.waveform_metadata),
        DF_ATTR: catalog.df,
        **catalog.population.to_attrs(),
        POPULATION_NUM_SAMPLES_ATTR: catalog.num_samples,
        POPULATION_PARAMS_ATTR: json.dumps(dict(catalog.fiducials), sort_keys=True),
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
        attrs = decoded_attrs(handle)
        names = json_array_attr(
            attrs[PARAMETER_NAMES_ATTR], label=label, name=PARAMETER_NAMES_ATTR
        )
        population = PopulationRecord.from_attrs(attrs, label=label)
        catalog = cls(
            source_parameters=unstack_columns(
                np.asarray(handle["source_parameters"]), names
            ),
            polarization_power=np.asarray(handle["polarization_power"]),
            frequencies=np.asarray(handle["frequency"]),
            waveform_metadata=waveform_from_attrs(
                attrs, label=label, frequency_resolution_fallback=DF_ATTR
            ),
            _population=population,
            _fiducials={
                name: float(value)
                for name, value in json_object_attr(
                    attrs[POPULATION_PARAMS_ATTR],
                    label=label,
                    name=POPULATION_PARAMS_ATTR,
                ).items()
            },
        )
    population.check_registered()
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
    waveform_from_attrs(
        decoded_attrs(handle), label=label, frequency_resolution_fallback=DF_ATTR
    )
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

    require_attrs(
        handle.attrs,
        REQUIRED_CATALOG_ATTRS,
        label=label,
        kind="population metadata",
        hint="; regenerate this catalog",
    )
    count = int_attr(
        handle.attrs[POPULATION_NUM_SAMPLES_ATTR],
        label=label,
        name=POPULATION_NUM_SAMPLES_ATTR,
    )
    if count != source.shape[0]:
        raise ValueError(
            f"{label}: population_num_samples ({count}) does not match the sample "
            f"dimension ({source.shape[0]})"
        )
    names = _column_names(handle.attrs, label=label, columns=source.shape[1])
    if REDSHIFT_SITE not in names:
        raise ValueError(f"{label}: missing the {REDSHIFT_SITE!r} source parameter")


# --------------------------------------------------------------------- #
# Spectral-density catalogs
# --------------------------------------------------------------------- #
SPECTRAL_DENSITY_FORMAT_NAME = "astrogwb_spectral_density_v2"
SPECTRAL_DENSITY_DATASETS = (
    "frequency",
    "spectral_density",
    "n_events",
    "total_merger_rate",
    "hyperparameters",
)

N_MAX_SIGMA_ATTR = "n_max_sigma"
OBSERVATION_TIME_ATTR = "observation_time"

#: What a spectra file must carry. Unlike the catalog format there is no
#: optional tier -- v1 is the first version, so every population attribute is
#: required outright, and a field added later gets its own additive treatment
#: then.
REQUIRED_SPECTRAL_DENSITY_ATTRS = (
    *POPULATION_ATTRS,
    N_MAX_SIGMA_ATTR,
    OBSERVATION_TIME_ATTR,
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
    attrs: dict[str, str | int | float] = {
        FORMAT_NAME_ATTR: SPECTRAL_DENSITY_FORMAT_NAME,
        "domain": DOMAIN_FREQUENCY,
        **waveform_attrs(catalog.waveform_metadata),
        **catalog.population.to_attrs(),
        PARAMETER_NAMES_ATTR: json.dumps(names),
        N_MAX_SIGMA_ATTR: catalog.n_max_sigma,
        OBSERVATION_TIME_ATTR: catalog.observation_time,
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
        attrs = decoded_attrs(handle)
        names = json_array_attr(
            attrs[PARAMETER_NAMES_ATTR], label=label, name=PARAMETER_NAMES_ATTR
        )
        population = PopulationRecord.from_attrs(attrs, label=label)
        catalog = cls(
            spectral_density=np.asarray(handle["spectral_density"]),
            frequencies=np.asarray(handle["frequency"]),
            n_events=np.asarray(handle["n_events"]),
            total_merger_rate=np.asarray(handle["total_merger_rate"]),
            hyperparameters=unstack_columns(
                np.asarray(handle["hyperparameters"]), names
            ),
            waveform_metadata=waveform_from_attrs(attrs, label=label),
            _population=population,
            n_max_sigma=float(attrs[N_MAX_SIGMA_ATTR]),
            observation_time=float(attrs[OBSERVATION_TIME_ATTR]),
        )
    population.check_registered()
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
    waveform_from_attrs(decoded_attrs(handle), label=label)
    require_attrs(
        handle.attrs,
        REQUIRED_SPECTRAL_DENSITY_ATTRS,
        label=label,
        kind="population metadata",
    )

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
