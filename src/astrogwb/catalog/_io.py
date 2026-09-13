"""Direct HDF5 persistence for :class:`~astrogwb.catalog.Catalog`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from astrogwb.catalog.catalog import REDSHIFT_SITE, Catalog
from astrogwb.waveform import PolarizationPowerGenerator

try:
    import h5py
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "astrogwb.catalog file I/O needs h5py. Install it with the 'io' extra: "
        "pip install 'astrogwb[io]'"
    ) from error

__all__ = ["DOMAIN_FREQUENCY", "FORMAT_NAME", "load_catalog", "save_catalog"]

FORMAT_NAME = "astrogwb_catalog_v5"
DOMAIN_FREQUENCY = "frequency"
WAVEFORM_ATTRS = (
    "approximant",
    "minimum_frequency",
    "maximum_frequency",
    "reference_frequency",
    "sampling_frequency",
    "df",
)
#: Written on every save, but optional on read: older v5 files predate it, and
#: for every file written so far it equals the ``df`` attribute (see
#: ``waveform_metadata_from_file``).
FREQUENCY_RESOLUTION_ATTR = "frequency_resolution"
POPULATION_SEED_ATTR = "population_seed"
POPULATION_NUM_SAMPLES_ATTR = "population_num_samples"
MODEL_NAME_ATTR = "population_model"
#: Written on every save, additive on top of the legacy single ``MODEL_NAME_ATTR``.
#: Optional on read: a v5 file written before the source/rate split has only
#: ``MODEL_NAME_ATTR``, treated as the source-model name (see ``load_catalog``).
SOURCE_MODEL_NAME_ATTR = "population_source_model"
RATE_MODEL_NAME_ATTR = "population_rate_model"
#: The rate every pre-split v5 catalog was drawn at. Those files name only
#: the source (as ``population_model``); this is the pairing they all used.
_LEGACY_RATE_MODEL_NAME = "madau_dickinson"
MODEL_KWARGS_ATTR = "population_model_kwargs"
POPULATION_PARAMS_ATTR = "population_params"
DENSITY_SITES_ATTR = "population_density_sites"
PARAMETER_NAMES_ATTR = "source_parameter_names"
POPULATION_ATTRS = (
    POPULATION_SEED_ATTR,
    POPULATION_NUM_SAMPLES_ATTR,
    MODEL_NAME_ATTR,
    MODEL_KWARGS_ATTR,
    POPULATION_PARAMS_ATTR,
    DENSITY_SITES_ATTR,
)
REQUIRED_POPULATION_ATTRS = POPULATION_ATTRS


def save_catalog(
    catalog: Catalog, path: str | Path, *, compression: str | None = None
) -> None:
    """Write one catalog in the v5 direct-HDF5 format."""
    waveform = catalog.waveform_metadata
    names = list(catalog.source_parameters)
    source_parameters = (
        np.stack(
            [
                np.asarray(catalog.source_parameters[name], dtype=np.float64)
                for name in names
            ],
            axis=1,
        )
        if names
        else np.empty((catalog.num_samples, 0), dtype=np.float64)
    )
    attrs: dict[str, str | int | float] = {
        "format_name": FORMAT_NAME,
        "domain": DOMAIN_FREQUENCY,
        "approximant": waveform.approximant,
        "minimum_frequency": waveform.minimum_frequency,
        "maximum_frequency": waveform.maximum_frequency,
        "reference_frequency": waveform.reference_frequency,
        "sampling_frequency": waveform.sampling_frequency,
        # Informational only, measured from the `frequency` dataset on write
        # and never read back into a computation -- the dataset is the truth.
        "df": catalog.df,
        FREQUENCY_RESOLUTION_ATTR: waveform.frequency_resolution,
        POPULATION_SEED_ATTR: catalog.seed,
        POPULATION_NUM_SAMPLES_ATTR: catalog.num_samples,
        MODEL_NAME_ATTR: catalog.population_model_name,
        SOURCE_MODEL_NAME_ATTR: catalog.population_source_model_name,
        RATE_MODEL_NAME_ATTR: catalog.population_rate_model_name,
        MODEL_KWARGS_ATTR: json.dumps(catalog.population_model_kwargs, sort_keys=True),
        POPULATION_PARAMS_ATTR: json.dumps(catalog.fiducials, sort_keys=True),
        DENSITY_SITES_ATTR: json.dumps(list(catalog.density_sites)),
        PARAMETER_NAMES_ATTR: json.dumps(names),
    }
    with h5py.File(path, "w") as handle:
        for name, value in attrs.items():
            handle.attrs[name] = value
        handle.create_dataset(
            "frequency", data=np.asarray(catalog.frequencies), compression=compression
        )
        handle.create_dataset(
            "polarization_power",
            data=np.asarray(catalog.polarization_power),
            compression=compression,
        )
        handle.create_dataset(
            "source_parameters", data=source_parameters, compression=compression
        )


def load_catalog[C: Catalog](cls: type[C], path: str | Path) -> C:
    """Read and structurally validate one catalog, rebuilding its model record."""
    label = Path(path).name
    with h5py.File(path, "r") as handle:
        validate_catalog_file(handle, label=label)
        waveform = waveform_metadata_from_file(handle, label=label)
        attrs = {
            str(name): _scalar(value, name=str(name))
            for name, value in handle.attrs.items()
        }
        names = _json_list(
            attrs[PARAMETER_NAMES_ATTR], label=label, name=PARAMETER_NAMES_ATTR
        )
        seed = attrs[POPULATION_SEED_ATTR]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"{label}: {POPULATION_SEED_ATTR} must be an int")
        if SOURCE_MODEL_NAME_ATTR in attrs and RATE_MODEL_NAME_ATTR in attrs:
            source_model_name = str(attrs[SOURCE_MODEL_NAME_ATTR])
            rate_model_name = str(attrs[RATE_MODEL_NAME_ATTR])
        else:
            # A v5 file written before the source/rate split: the single
            # ``population_model`` attribute is the source-model name, and
            # every such catalog used the Madau-Dickinson rate.
            source_model_name = str(attrs[MODEL_NAME_ATTR])
            rate_model_name = _LEGACY_RATE_MODEL_NAME
        values = np.asarray(handle["source_parameters"])
        catalog = cls(
            source_parameters={
                str(name): values[:, index] for index, name in enumerate(names)
            },
            polarization_power=np.asarray(handle["polarization_power"]),
            frequencies=np.asarray(handle["frequency"]),
            waveform_metadata=waveform,
            _source_model_name=source_model_name,
            _rate_model_name=rate_model_name,
            _model_kwargs=_json_mapping(
                attrs[MODEL_KWARGS_ATTR], label=label, name=MODEL_KWARGS_ATTR
            ),
            _fiducials={
                name: float(value)
                for name, value in _json_mapping(
                    attrs[POPULATION_PARAMS_ATTR],
                    label=label,
                    name=POPULATION_PARAMS_ATTR,
                ).items()
            },
            _density_sites=tuple(
                _json_list(
                    attrs[DENSITY_SITES_ATTR], label=label, name=DENSITY_SITES_ATTR
                )
            ),
            seed=seed,
        )
    # verify registry reconstruction
    catalog.get_source_model()
    catalog.get_merger_rate_fn()
    return catalog


def validate_catalog_file(handle: h5py.File | h5py.Group, *, label: str) -> None:
    """Validate HDF5 layout, metadata, shapes, and serialized dtypes."""
    _check_format(handle.attrs, label=label)
    for name in ("frequency", "polarization_power", "source_parameters"):
        if name not in handle:
            raise ValueError(f"{label}: missing '{name}' dataset")
    frequency = handle["frequency"]
    if frequency.ndim != 1:
        raise ValueError(f"{label}: frequency dataset must be one-dimensional")
    waveform_metadata_from_file(handle, label=label)
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
    missing = [name for name in REQUIRED_POPULATION_ATTRS if name not in handle.attrs]
    if missing:
        raise ValueError(
            f"{label}: missing population metadata attribute(s): {', '.join(missing)}; regenerate this catalog"
        )
    count = _scalar(
        handle.attrs[POPULATION_NUM_SAMPLES_ATTR], name=POPULATION_NUM_SAMPLES_ATTR
    )
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(f"{label}: population_num_samples must be an int")
    if count != source.shape[0]:
        raise ValueError(
            f"{label}: population_num_samples ({count}) does not match the sample dimension ({source.shape[0]})"
        )
    names = _json_list(
        handle.attrs.get(PARAMETER_NAMES_ATTR, ""),
        label=label,
        name=PARAMETER_NAMES_ATTR,
    )
    if len(names) != source.shape[1] or len(set(names)) != len(names):
        raise ValueError(
            f"{label}: source parameter ordering does not match source_parameters"
        )
    if REDSHIFT_SITE not in names:
        raise ValueError(f"{label}: missing the {REDSHIFT_SITE!r} source parameter")


def waveform_metadata_from_file(
    handle: h5py.Group, *, label: str
) -> PolarizationPowerGenerator:
    missing = [name for name in WAVEFORM_ATTRS if name not in handle.attrs]
    if missing:
        raise ValueError(
            f"{label}: missing waveform metadata attribute(s): {', '.join(missing)}"
        )
    # `frequency_resolution` is additive: a pre-existing v5 file predates it,
    # and falls back to `df`, since for every file written so far the two are
    # equal. This is not a cross-check -- df is never read back into a
    # computation -- just a reasonable default for an older file's *request*.
    if FREQUENCY_RESOLUTION_ATTR in handle.attrs:
        frequency_resolution = _scalar(
            handle.attrs[FREQUENCY_RESOLUTION_ATTR], name=FREQUENCY_RESOLUTION_ATTR
        )
    else:
        frequency_resolution = handle.attrs["df"]
    try:
        return PolarizationPowerGenerator(
            approximant=str(_scalar(handle.attrs["approximant"], name="approximant")),
            minimum_frequency=float(
                _scalar(handle.attrs["minimum_frequency"], name="minimum_frequency")
            ),
            maximum_frequency=float(
                _scalar(handle.attrs["maximum_frequency"], name="maximum_frequency")
            ),
            reference_frequency=float(
                _scalar(handle.attrs["reference_frequency"], name="reference_frequency")
            ),
            sampling_frequency=float(
                _scalar(handle.attrs["sampling_frequency"], name="sampling_frequency")
            ),
            frequency_resolution=float(frequency_resolution),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid waveform metadata: {error}") from error


def _json_mapping(value: Any, *, label: str, name: str) -> dict[str, Any]:
    decoded = _json(value, label=label, name=name)
    if not isinstance(decoded, dict):
        raise TypeError(f"{label}: attribute {name!r} must decode to a JSON object")
    return decoded


def _json_list(value: Any, *, label: str, name: str) -> list[Any]:
    decoded = _json(value, label=label, name=name)
    if not isinstance(decoded, list):
        raise TypeError(f"{label}: attribute {name!r} must decode to a JSON array")
    return decoded


def _json(value: Any, *, label: str, name: str) -> Any:
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label}: attribute {name!r} is not valid JSON: {error}"
        ) from error


def _check_format(attrs: Mapping[Any, Any], *, label: str) -> None:
    if attrs.get("format_name") != FORMAT_NAME:
        raise ValueError(
            f"{label}: format_name is {attrs.get('format_name')!r}, expected {FORMAT_NAME!r}"
        )
    if attrs.get("domain") != DOMAIN_FREQUENCY:
        raise ValueError(
            f"{label}: domain is {attrs.get('domain')!r}, expected {DOMAIN_FREQUENCY!r}"
        )


def _scalar(value: Any, *, name: str) -> str | int | float:
    if isinstance(value, bytes):
        return value.decode()
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
