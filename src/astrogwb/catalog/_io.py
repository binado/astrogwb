"""Private xarray/HDF5 encoding for :class:`~astrogwb.catalog.Catalog`.

xarray pulls in pandas, which the publishable wheel deliberately does not
carry, so this module lives behind the ``io`` optional dependency. It is
imported inside :meth:`Catalog.load` and :meth:`Catalog.save` rather than at
module scope, which is what keeps an in-memory catalog usable without those
extras installed.

netCDF attributes are flat scalars, so the population record travels as three
JSON strings beside the plain ``population_model`` name. What is *not* stored
is a callable: the model is rebuilt from the registry on load.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from astrogwb.catalog.catalog import Catalog
from astrogwb.populations import (
    REDSHIFT_SITE,
    redshift_log_density,
)
from astrogwb.waveform import PolarizationPowerGenerator

try:
    import xarray as xr
except ImportError as error:  # pragma: no cover - depends on the install extras
    raise ImportError(
        "astrogwb.catalog file I/O needs xarray and h5netcdf, which are not core "
        "dependencies. Install them with the 'io' extra: "
        "pip install 'astrogwb[io]'."
    ) from error

__all__ = [
    "DOMAIN_FREQUENCY",
    "FORMAT_NAME",
    "catalog_from_dataset",
    "catalog_to_dataset",
    "load_catalog",
    "save_catalog",
]

#: Version 3 records ordered included density sites. Earlier formats require
#: regeneration; no excluded-factor compatibility reader is provided.
FORMAT_NAME = "astrogwb_catalog_v3"
LEGACY_FORMAT_NAMES = ("waveform_catalog", "astrogwb_catalog", "astrogwb_catalog_v2")
DOMAIN_FREQUENCY = "frequency"

WAVEFORM_ATTRS = (
    "approximant",
    "minimum_frequency",
    "maximum_frequency",
    "reference_frequency",
    "sampling_frequency",
    "df",
)
POPULATION_SEED_ATTR = "population_seed"
POPULATION_NUM_SAMPLES_ATTR = "population_num_samples"
MODEL_NAME_ATTR = "population_model"
MODEL_KWARGS_ATTR = "population_model_kwargs"
POPULATION_PARAMS_ATTR = "population_params"
DENSITY_SITES_ATTR = "population_density_sites"

#: Kept from the previous format, demoted from source of truth to assertion.
#: See :func:`_redshift_density_probe`.
PROPOSAL_ATTR = "redshift_proposal"

POPULATION_ATTRS = (
    POPULATION_SEED_ATTR,
    POPULATION_NUM_SAMPLES_ATTR,
    MODEL_NAME_ATTR,
    MODEL_KWARGS_ATTR,
    POPULATION_PARAMS_ATTR,
    DENSITY_SITES_ATTR,
    PROPOSAL_ATTR,
)

#: The population record is mandatory: a file missing any of these cannot say
#: what density drew it, and no amount of inference from the run config is an
#: acceptable substitute for that.
REQUIRED_POPULATION_ATTRS = POPULATION_ATTRS

#: Number of interior redshifts the drift fingerprint is evaluated at.
PROBE_POINTS = 8

#: Tolerances for the two load-time consistency checks. Both compare a value
#: recomputed now against one computed at generation time, which ran the same
#: expressions under ``vmap``; agreement is to floating-point noise, not to the
#: last bit.
DERIVED_COLUMN_RTOL = 1e-9
DERIVED_COLUMN_ATOL = 1e-12


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #
def catalog_to_dataset(catalog: Catalog) -> xr.Dataset:
    """Encode a catalog, including its complete population record."""
    waveform = catalog.waveform_metadata
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
        source_parameters = np.empty((catalog.num_samples, 0), dtype=np.float64)

    attrs: dict[str, str | int | float] = {
        "format_name": FORMAT_NAME,
        "domain": DOMAIN_FREQUENCY,
        "approximant": waveform.approximant,
        "minimum_frequency": waveform.minimum_frequency,
        "maximum_frequency": waveform.maximum_frequency,
        "reference_frequency": waveform.reference_frequency,
        "sampling_frequency": waveform.sampling_frequency,
        "df": waveform.df,
        POPULATION_SEED_ATTR: catalog.seed,
        POPULATION_NUM_SAMPLES_ATTR: catalog.num_samples,
        MODEL_NAME_ATTR: catalog.population_model_name,
        MODEL_KWARGS_ATTR: json.dumps(catalog.population_model_kwargs, sort_keys=True),
        POPULATION_PARAMS_ATTR: json.dumps(catalog.fiducials, sort_keys=True),
        DENSITY_SITES_ATTR: json.dumps(list(catalog.density_sites)),
        PROPOSAL_ATTR: json.dumps(_redshift_density_probe(catalog)),
    }

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
            "frequency": np.asarray(waveform.frequencies),
            "parameter": names,
        },
        attrs=attrs,
    )
    validate_catalog_dataset(dataset, label="catalog")
    return dataset


def save_catalog(
    catalog: Catalog, path: str | Path, *, compression: str | None = None
) -> None:
    """Write a catalog to the astrogwb HDF5 format."""
    dataset = catalog_to_dataset(catalog)
    encoding = (
        {"polarization_power": {"compression": compression}}
        if compression is not None
        else None
    )
    dataset.to_netcdf(path, engine="h5netcdf", encoding=encoding)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def load_catalog[C: Catalog](cls: type[C], path: str | Path) -> C:
    """Read, decode, and fully validate one catalog file."""
    label = Path(path).name
    dataset = xr.load_dataset(path, engine="h5netcdf")
    catalog = catalog_from_dataset(dataset, cls=cls, label=label)
    check_population_consistency(
        catalog,
        recorded_probe=_decode_probe(dataset.attrs, label=label),
        label=label,
    )
    return catalog


def catalog_from_dataset[C: Catalog](
    dataset: xr.Dataset, *, cls: type[C], label: str = "catalog"
) -> C:
    """Decode a validated Dataset into a catalog, without running its model."""
    validate_catalog_dataset(dataset, label=label)
    waveform = waveform_metadata_from_dataset(dataset, label=label)
    decoded = {
        str(name): _scalar(value, name=str(name))
        for name, value in dataset.attrs.items()
    }
    seed = decoded[POPULATION_SEED_ATTR]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"{label}: population_seed must be an int")
    names = [str(name) for name in dataset.coords["parameter"].values.tolist()]
    parameters = {
        name: np.asarray(dataset["source_parameters"].isel(parameter=index).values)
        for index, name in enumerate(names)
    }
    return cls(
        source_parameters=parameters,
        polarization_power=np.asarray(dataset["polarization_power"].values),
        waveform_metadata=waveform,
        _model_name=str(decoded[MODEL_NAME_ATTR]),
        _model_kwargs=_json_mapping(
            decoded[MODEL_KWARGS_ATTR], label=label, name=MODEL_KWARGS_ATTR
        ),
        _fiducials={
            name: float(value)
            for name, value in _json_mapping(
                decoded[POPULATION_PARAMS_ATTR],
                label=label,
                name=POPULATION_PARAMS_ATTR,
            ).items()
        },
        _density_sites=tuple(
            _json_list(
                decoded[DENSITY_SITES_ATTR], label=label, name=DENSITY_SITES_ATTR
            )
        ),
        seed=seed,
    )


def check_population_consistency(
    catalog: Catalog, *, recorded_probe: Mapping[str, list[float]], label: str
) -> None:
    """Prove the recorded population still describes the stored arrays.

    Two independent checks, because they fail for different reasons:

    - The **drift guard** recomputes the redshift log density at the probe
      points recorded when the file was written. A registry key pins a name,
      not the mathematics behind it, so this is what catches a registered model
      whose density changed underneath an existing catalog.
    - The **derived-column check** re-executes the model at the stored
      stochastic values and compares every deterministic it declares against
      the stored column. This is what catches columns that were computed by
      some other route and have since drifted.
    """
    model = catalog.get_population_model()
    params = catalog.fiducials

    recomputed = np.asarray(
        redshift_log_density(model, params, np.asarray(recorded_probe["redshift"]))
    )
    expected = np.asarray(recorded_probe["log_prob"], dtype=np.float64)
    for index, (probe, want, got) in enumerate(
        zip(recorded_probe["redshift"], expected, recomputed, strict=True)
    ):
        if not np.isclose(want, got, rtol=DERIVED_COLUMN_RTOL, atol=0.0):
            raise ValueError(
                f"{label}: population model {catalog.population_model_name!r} no "
                f"longer reproduces the redshift density this catalog was drawn "
                f"from: log p(z={probe:.6g}) recorded {want:.12g}, recomputed "
                f"{got:.12g} (probe {index}). Regenerate the catalog, or restore "
                "the registered model."
            )

    values = catalog.source_parameters
    _, trace = model.evaluate(params, values)
    for name in sorted(catalog.source_parameters):
        if name not in trace or trace[name]["type"] != "deterministic":
            continue
        stored = np.asarray(catalog.source_parameters[name], dtype=np.float64)
        derived = np.asarray(trace[name]["value"], dtype=np.float64)
        if not np.allclose(
            stored, derived, rtol=DERIVED_COLUMN_RTOL, atol=DERIVED_COLUMN_ATOL
        ):
            worst = int(np.argmax(np.abs(stored - derived)))
            raise ValueError(
                f"{label}: stored column {name!r} disagrees with the value "
                f"population model {catalog.population_model_name!r} derives from "
                f"the stored source samples: sample {worst} holds "
                f"{stored[worst]:.12g}, the model gives {derived[worst]:.12g}. "
                "Regenerate the catalog."
            )


# --------------------------------------------------------------------------- #
# Structural validation
# --------------------------------------------------------------------------- #
def validate_catalog_dataset(dataset: xr.Dataset, *, label: str) -> None:
    """Validate the catalog format without running the population model."""
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
    if REDSHIFT_SITE not in parameter_names:
        raise ValueError(f"{label}: missing the {REDSHIFT_SITE!r} source parameter")

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

    missing = [name for name in REQUIRED_POPULATION_ATTRS if name not in dataset.attrs]
    if missing:
        raise ValueError(
            f"{label}: missing population metadata attribute(s): "
            f"{', '.join(missing)}; regenerate this catalog"
        )
    num_samples_scalar = _scalar(
        dataset.attrs[POPULATION_NUM_SAMPLES_ATTR], name=POPULATION_NUM_SAMPLES_ATTR
    )
    if isinstance(num_samples_scalar, bool) or not isinstance(num_samples_scalar, int):
        raise TypeError(f"{label}: population_num_samples must be an int")
    if dataset.sizes["sample"] != num_samples_scalar:
        raise ValueError(
            f"{label}: population_num_samples ({num_samples_scalar}) does not "
            f"match the sample dimension ({dataset.sizes['sample']})"
        )


def waveform_metadata_from_dataset(
    dataset: xr.Dataset, *, label: str
) -> PolarizationPowerGenerator:
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
        return PolarizationPowerGenerator(
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _redshift_density_probe(catalog: Catalog) -> dict[str, list[float]]:
    """Fingerprint the generating redshift density at fixed interior probes.

    Interior points only: the endpoints of the generation window sit on the
    edge of the interpolation table, where the density is zero and the log is
    ``-inf`` -- a probe that carries no information and does not round-trip
    through JSON.
    """
    kwargs = catalog.population_model_kwargs
    z_min = float(kwargs.get("z_min", 0.0))
    z_max = float(kwargs.get("z_max", 1.0))
    probes = np.linspace(z_min, z_max, PROBE_POINTS + 2)[1:-1]
    log_prob = np.asarray(
        redshift_log_density(catalog.get_population_model(), catalog.fiducials, probes),
        dtype=np.float64,
    )
    return {
        "redshift": [float(value) for value in probes],
        "log_prob": [float(value) for value in log_prob],
    }


def _decode_probe(attrs: Mapping[Any, Any], *, label: str) -> dict[str, list[float]]:
    raw = attrs.get(PROPOSAL_ATTR)
    if raw is None:
        raise ValueError(
            f"{label}: missing the {PROPOSAL_ATTR!r} drift fingerprint; "
            "regenerate this catalog"
        )
    probe = _json_mapping(raw, label=label, name=PROPOSAL_ATTR)
    if set(probe) != {"redshift", "log_prob"}:
        raise ValueError(
            f"{label}: {PROPOSAL_ATTR!r} must hold 'redshift' and 'log_prob' lists"
        )
    return {name: [float(value) for value in probe[name]] for name in probe}


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
    format_name = attrs.get("format_name")
    if format_name in LEGACY_FORMAT_NAMES:
        raise ValueError(
            f"{label}: format_name={format_name!r} predates the catalog population "
            "record and carries no reconstructable source density; regenerate this "
            "catalog with scripts/generate_catalog.py"
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
