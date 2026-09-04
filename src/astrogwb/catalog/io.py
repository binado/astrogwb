"""JAX-free xarray and HDF5 serialization for array-native catalogs.

xarray pulls in pandas, which the publishable wheel deliberately does not
carry, so this module lives behind the ``io`` optional dependency rather than
in the core dependency set.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from astrogwb.catalog import (
    Catalog,
    FrequencyDomainWaveformMetadata,
    PopulationMetadata,
)

try:
    import xarray as xr
except ImportError as error:  # pragma: no cover - depends on the install extras
    raise ImportError(
        "astrogwb.catalog.io needs xarray and h5netcdf, which are not core "
        "dependencies. Install them with the 'io' extra: "
        "pip install 'astrogwb[io]'."
    ) from error

__all__ = [
    "DOMAIN_FREQUENCY",
    "FORMAT_NAME",
    "RESERVED_ATTRS",
    "catalog_from_dataset",
    "catalog_to_dataset",
    "load_catalog",
    "open_catalog",
    "population_metadata_from_attrs",
    "save_catalog",
    "validate_catalog_dataset",
    "waveform_metadata_from_dataset",
]

FORMAT_NAME = "astrogwb_catalog"
LEGACY_FORMAT_NAME = "waveform_catalog"
DOMAIN_FREQUENCY = "frequency"

WAVEFORM_ATTRS = (
    "approximant",
    "minimum_frequency",
    "maximum_frequency",
    "reference_frequency",
    "sampling_frequency",
    "df",
)
POPULATION_NAME_ATTR = "population_name"
POPULATION_SEED_ATTR = "population_seed"
POPULATION_NUM_SAMPLES_ATTR = "population_num_samples"
POPULATION_SOURCE_TYPE_ATTR = "population_source_type"
POPULATION_ATTRS = (
    POPULATION_NAME_ATTR,
    POPULATION_SEED_ATTR,
    POPULATION_NUM_SAMPLES_ATTR,
    POPULATION_SOURCE_TYPE_ATTR,
)
RESERVED_ATTRS = frozenset(
    {"format_name", "domain", *WAVEFORM_ATTRS, *POPULATION_ATTRS}
)


def catalog_to_dataset(catalog: Catalog) -> xr.Dataset:
    """Encode a catalog in the stacked xarray format."""
    waveform = catalog.waveform_metadata
    population = catalog.population_metadata
    collisions = sorted(RESERVED_ATTRS.intersection(population.provenance))
    if collisions:
        raise ValueError(
            "population provenance may not override reserved catalog attribute(s): "
            + ", ".join(collisions)
        )

    names = list(catalog.source_parameters)
    if names:
        source_parameters = np.stack(
            [
                np.asarray(catalog.source_parameters[name], dtype=np.float64)
                for name in names
            ],
            axis=1,
        )
    else:
        source_parameters = np.empty((population.num_samples, 0), dtype=np.float64)

    attrs: dict[str, str | int | float] = {
        "format_name": FORMAT_NAME,
        "domain": DOMAIN_FREQUENCY,
        "approximant": waveform.approximant,
        "minimum_frequency": waveform.minimum_frequency,
        "maximum_frequency": waveform.maximum_frequency,
        "reference_frequency": waveform.reference_frequency,
        "sampling_frequency": waveform.sampling_frequency,
        "df": waveform.df,
        POPULATION_NAME_ATTR: population.name,
        POPULATION_SEED_ATTR: population.seed,
        POPULATION_NUM_SAMPLES_ATTR: population.num_samples,
        **population.provenance,
    }
    if population.source_type is not None:
        attrs[POPULATION_SOURCE_TYPE_ATTR] = population.source_type

    dataset = xr.Dataset(
        data_vars={
            "polarization_power": (
                ("frequency", "sample"),
                catalog.polarization_power,
            ),
            "source_parameters": (
                ("sample", "parameter"),
                source_parameters,
            ),
        },
        coords={
            "frequency": waveform.frequencies,
            "parameter": names,
        },
        attrs=attrs,
    )
    validate_catalog_dataset(dataset, label="catalog")
    return dataset


def catalog_from_dataset(dataset: xr.Dataset) -> Catalog:
    """Decode a validated stacked Dataset into an array-native core catalog."""
    validate_catalog_dataset(dataset, label="catalog")
    waveform = waveform_metadata_from_dataset(dataset, label="catalog")
    population = population_metadata_from_attrs(dataset.attrs, label="catalog")
    names = [str(name) for name in dataset.coords["parameter"].values.tolist()]
    parameters = {
        name: np.asarray(dataset["source_parameters"].isel(parameter=index).values)
        for index, name in enumerate(names)
    }
    return Catalog(
        source_parameters=parameters,
        polarization_power=np.asarray(dataset["polarization_power"].values),
        waveform_metadata=waveform,
        population_metadata=population,
    )


def save_catalog(
    path: str | Path,
    catalog: Catalog,
    *,
    compression: str | None = None,
) -> None:
    """Write a catalog to the astrogwb HDF5 format."""
    dataset = catalog_to_dataset(catalog)
    encoding = (
        {"polarization_power": {"compression": compression}}
        if compression is not None
        else None
    )
    dataset.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def load_catalog(path: str | Path) -> xr.Dataset:
    """Load and validate a catalog Dataset eagerly."""
    label = Path(path).name
    dataset = xr.load_dataset(path, engine="h5netcdf")
    validate_catalog_dataset(dataset, label=label)
    return dataset


def open_catalog(path: str | Path) -> xr.Dataset:
    """Open and validate a catalog while leaving its data arrays lazy."""
    label = Path(path).name
    dataset = xr.open_dataset(path, engine="h5netcdf")
    try:
        validate_catalog_dataset(dataset, label=label)
    except Exception:
        dataset.close()
        raise
    return dataset


def validate_catalog_dataset(dataset: xr.Dataset, *, label: str) -> None:
    """Validate the catalog format without loading polarization power."""
    _check_format(dataset.attrs, label=label)

    if "frequency" not in dataset.coords:
        raise ValueError(f"{label}: missing 'frequency' coordinate")
    if dataset.coords["frequency"].dims != ("frequency",):
        raise ValueError(f"{label}: frequency coordinate must be one-dimensional")
    waveform_metadata_from_dataset(dataset, label=label)

    if "parameter" not in dataset.coords:
        raise ValueError(f"{label}: missing 'parameter' coordinate")
    if dataset.coords["parameter"].dims != ("parameter",):
        raise ValueError(f"{label}: parameter coordinate must be one-dimensional")
    parameter_names = [
        str(name) for name in dataset.coords["parameter"].values.tolist()
    ]
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError(f"{label}: parameter names must be unique")

    if "polarization_power" not in dataset:
        raise ValueError(f"{label}: missing 'polarization_power' data variable")
    power = dataset["polarization_power"]
    if power.dims != ("frequency", "sample"):
        raise ValueError(
            f"{label}: 'polarization_power' must have dims (frequency, sample), "
            f"got {power.dims}"
        )
    if not np.issubdtype(power.dtype, np.number) or np.issubdtype(
        power.dtype, np.complexfloating
    ):
        raise ValueError(f"{label}: 'polarization_power' must be real-valued")

    if "source_parameters" not in dataset:
        raise ValueError(f"{label}: missing 'source_parameters' data variable")
    source_parameters = dataset["source_parameters"]
    if source_parameters.dims != ("sample", "parameter"):
        raise ValueError(
            f"{label}: 'source_parameters' must have dims (sample, parameter), "
            f"got {source_parameters.dims}"
        )
    if source_parameters.dtype != np.dtype(np.float64):
        raise ValueError(f"{label}: 'source_parameters' must be serialized as float64")

    population = population_metadata_from_attrs(dataset.attrs, label=label)
    if dataset.sizes["sample"] != population.num_samples:
        raise ValueError(
            f"{label}: population_num_samples ({population.num_samples}) does not "
            f"match the sample dimension ({dataset.sizes['sample']})"
        )


def waveform_metadata_from_dataset(
    dataset: xr.Dataset, *, label: str
) -> FrequencyDomainWaveformMetadata:
    """Decode waveform metadata and the coordinate without touching data variables."""
    missing = [name for name in WAVEFORM_ATTRS if name not in dataset.attrs]
    if missing:
        raise ValueError(
            f"{label}: missing waveform metadata attribute(s): {', '.join(missing)}"
        )
    if "frequency" not in dataset.coords:
        raise ValueError(f"{label}: missing 'frequency' coordinate")
    attrs = dataset.attrs
    try:
        return FrequencyDomainWaveformMetadata(
            frequencies=np.asarray(dataset.coords["frequency"].values),
            approximant=str(_scalar(attrs["approximant"], name="approximant")),
            minimum_frequency=float(
                _scalar(attrs["minimum_frequency"], name="minimum_frequency")
            ),
            maximum_frequency=float(
                _scalar(attrs["maximum_frequency"], name="maximum_frequency")
            ),
            reference_frequency=float(
                _scalar(attrs["reference_frequency"], name="reference_frequency")
            ),
            sampling_frequency=float(
                _scalar(attrs["sampling_frequency"], name="sampling_frequency")
            ),
            df=float(_scalar(attrs["df"], name="df")),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid waveform metadata: {error}") from error


def population_metadata_from_attrs(
    attrs: Mapping[Any, Any], *, label: str
) -> PopulationMetadata:
    """Decode population metadata from attributes without loading catalog arrays."""
    required = (
        POPULATION_NAME_ATTR,
        POPULATION_SEED_ATTR,
        POPULATION_NUM_SAMPLES_ATTR,
    )
    missing = [name for name in required if name not in attrs]
    if missing:
        raise ValueError(
            f"{label}: missing population metadata attribute(s): {', '.join(missing)}; "
            "regenerate this catalog"
        )

    decoded = {
        str(name): _scalar(value, name=str(name)) for name, value in attrs.items()
    }
    seed = decoded[POPULATION_SEED_ATTR]
    num_samples = decoded[POPULATION_NUM_SAMPLES_ATTR]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"{label}: population_seed must be an int")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError(f"{label}: population_num_samples must be an int")

    provenance = {
        name: value for name, value in decoded.items() if name not in RESERVED_ATTRS
    }
    source_type_value = decoded.get(POPULATION_SOURCE_TYPE_ATTR)
    try:
        return PopulationMetadata(
            name=str(decoded[POPULATION_NAME_ATTR]),
            seed=seed,
            num_samples=num_samples,
            source_type=(None if source_type_value is None else str(source_type_value)),
            provenance=provenance,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid population metadata: {error}") from error


def _check_format(attrs: Mapping[Any, Any], *, label: str) -> None:
    format_name = attrs.get("format_name")
    if format_name == LEGACY_FORMAT_NAME:
        raise ValueError(
            f"{label}: format_name={LEGACY_FORMAT_NAME!r} is obsolete; regenerate "
            "this catalog with scripts/generate_bank.py"
        )
    if format_name != FORMAT_NAME:
        raise ValueError(
            f"{label}: format_name is {format_name!r}, expected {FORMAT_NAME!r}"
        )
    domain = attrs.get("domain")
    if domain != DOMAIN_FREQUENCY:
        raise ValueError(
            f"{label}: domain is {domain!r}, expected {DOMAIN_FREQUENCY!r}"
        )


def _scalar(value: Any, *, name: str) -> str | int | float:
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise TypeError(
            f"attribute {name!r} must be a str, non-boolean int, or float scalar"
        )
    return value
