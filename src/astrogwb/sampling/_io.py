"""Direct HDF5 persistence for spectrum-only simulation artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from astrogwb.gwb.spectral import AverageMode
from astrogwb.waveform import PolarizationPowerGenerator

try:
    import h5py
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "astrogwb.sampling file I/O needs h5py. Install it with the 'io' extra: "
        "pip install 'astrogwb[io]'"
    ) from error

__all__ = ["FORMAT_NAME", "SpectraArtifact", "load_spectra", "save_spectra"]

FORMAT_NAME = "astrogwb_spectra_v1"
DOMAIN_FREQUENCY = "frequency"
AVERAGE_MODES: tuple[AverageMode, ...] = (
    "analytic_inclination",
    "catalog_inclination",
)

WAVEFORM_ATTRS = (
    "approximant",
    "minimum_frequency",
    "maximum_frequency",
    "reference_frequency",
    "sampling_frequency",
    "frequency_resolution",
)
SOURCE_MODEL_NAME_ATTR = "population_source_model"
RATE_MODEL_NAME_ATTR = "population_rate_model"
MODEL_KWARGS_ATTR = "population_model_kwargs"
DENSITY_SITES_ATTR = "population_density_sites"
PARAMETER_NAMES_ATTR = "hyperparameter_names"
N_MAX_SIGMA_ATTR = "n_max_sigma"
SEED_ATTR = "population_seed"
AVERAGE_MODE_ATTR = "average_mode"
OBSERVATION_TIME_ATTR = "observation_time"

_REQUIRED_ATTRS = (
    SOURCE_MODEL_NAME_ATTR,
    RATE_MODEL_NAME_ATTR,
    MODEL_KWARGS_ATTR,
    DENSITY_SITES_ATTR,
    PARAMETER_NAMES_ATTR,
    N_MAX_SIGMA_ATTR,
    SEED_ATTR,
    AVERAGE_MODE_ATTR,
    OBSERVATION_TIME_ATTR,
)


class SpectraArtifact:
    """Spectrum-only simulation output in memory."""

    def __init__(
        self,
        *,
        frequencies: np.ndarray,
        spectral_density: np.ndarray,
        n_events: np.ndarray,
        total_merger_rate: np.ndarray,
        hyperparameters: Mapping[str, np.ndarray],
        source_model_name: str,
        rate_model_name: str,
        model_kwargs: Mapping[str, Any],
        density_sites: tuple[str, ...],
        waveform_metadata: PolarizationPowerGenerator,
        n_max_sigma: float,
        seed: int,
        average_mode: AverageMode,
        observation_time: float,
    ) -> None:
        self.frequencies = np.asarray(frequencies)
        self.spectral_density = np.asarray(spectral_density)
        self.n_events = np.asarray(n_events)
        self.total_merger_rate = np.asarray(total_merger_rate)
        self.hyperparameters = {
            name: np.asarray(values) for name, values in hyperparameters.items()
        }
        self.source_model_name = source_model_name
        self.rate_model_name = rate_model_name
        self.model_kwargs = dict(model_kwargs)
        self.density_sites = tuple(density_sites)
        self.waveform_metadata = waveform_metadata
        self.n_max_sigma = float(n_max_sigma)
        self.seed = int(seed)
        self.average_mode = _average_mode(average_mode)
        self.observation_time = float(observation_time)


def save_spectra(
    artifact: SpectraArtifact,
    path: str | Path,
    *,
    compression: str | None = None,
) -> None:
    """Write one spectrum-only artifact to HDF5."""
    names = list(artifact.hyperparameters)
    draws = artifact.spectral_density.shape[0]
    params = (
        np.stack(
            [
                np.asarray(artifact.hyperparameters[name], dtype=np.float64)
                for name in names
            ],
            axis=1,
        )
        if names
        else np.empty((draws, 0), dtype=np.float64)
    )

    waveform = artifact.waveform_metadata
    attrs: dict[str, str | int | float] = {
        "format_name": FORMAT_NAME,
        "domain": DOMAIN_FREQUENCY,
        "approximant": waveform.approximant,
        "minimum_frequency": waveform.minimum_frequency,
        "maximum_frequency": waveform.maximum_frequency,
        "reference_frequency": waveform.reference_frequency,
        "sampling_frequency": waveform.sampling_frequency,
        "frequency_resolution": waveform.frequency_resolution,
        SOURCE_MODEL_NAME_ATTR: artifact.source_model_name,
        RATE_MODEL_NAME_ATTR: artifact.rate_model_name,
        MODEL_KWARGS_ATTR: json.dumps(artifact.model_kwargs, sort_keys=True),
        DENSITY_SITES_ATTR: json.dumps(list(artifact.density_sites)),
        PARAMETER_NAMES_ATTR: json.dumps(names),
        N_MAX_SIGMA_ATTR: artifact.n_max_sigma,
        SEED_ATTR: artifact.seed,
        AVERAGE_MODE_ATTR: artifact.average_mode,
        OBSERVATION_TIME_ATTR: artifact.observation_time,
    }

    with h5py.File(path, "w") as handle:
        for name, value in attrs.items():
            handle.attrs[name] = value
        handle.create_dataset(
            "frequency", data=artifact.frequencies, compression=compression
        )
        handle.create_dataset(
            "spectral_density",
            data=artifact.spectral_density,
            compression=compression,
        )
        handle.create_dataset(
            "n_events", data=artifact.n_events, compression=compression
        )
        handle.create_dataset(
            "total_merger_rate",
            data=artifact.total_merger_rate,
            compression=compression,
        )
        handle.create_dataset("hyperparameters", data=params, compression=compression)


def load_spectra(path: str | Path) -> SpectraArtifact:
    """Read and validate one spectrum-only artifact."""
    label = Path(path).name
    with h5py.File(path, "r") as handle:
        validate_spectra_file(handle, label=label)
        attrs = {
            str(name): _scalar(value, name=str(name))
            for name, value in handle.attrs.items()
        }
        names = _json_list(
            attrs[PARAMETER_NAMES_ATTR], label=label, name=PARAMETER_NAMES_ATTR
        )
        values = np.asarray(handle["hyperparameters"])
        hyperparameters = {
            str(name): values[:, index] for index, name in enumerate(names)
        }
        seed = attrs[SEED_ATTR]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"{label}: {SEED_ATTR} must be an int")
        return SpectraArtifact(
            frequencies=np.asarray(handle["frequency"]),
            spectral_density=np.asarray(handle["spectral_density"]),
            n_events=np.asarray(handle["n_events"]),
            total_merger_rate=np.asarray(handle["total_merger_rate"]),
            hyperparameters=hyperparameters,
            source_model_name=str(attrs[SOURCE_MODEL_NAME_ATTR]),
            rate_model_name=str(attrs[RATE_MODEL_NAME_ATTR]),
            model_kwargs=_json_mapping(
                attrs[MODEL_KWARGS_ATTR], label=label, name=MODEL_KWARGS_ATTR
            ),
            density_sites=tuple(
                _json_list(
                    attrs[DENSITY_SITES_ATTR], label=label, name=DENSITY_SITES_ATTR
                )
            ),
            waveform_metadata=waveform_metadata_from_file(handle, label=label),
            n_max_sigma=float(attrs[N_MAX_SIGMA_ATTR]),
            seed=seed,
            average_mode=_average_mode(str(attrs[AVERAGE_MODE_ATTR])),
            observation_time=float(attrs[OBSERVATION_TIME_ATTR]),
        )


def validate_spectra_file(handle: h5py.File | h5py.Group, *, label: str) -> None:
    """Validate HDF5 layout, metadata, and array shapes."""
    _check_format(handle.attrs, label=label)
    for name in (
        "frequency",
        "spectral_density",
        "n_events",
        "total_merger_rate",
        "hyperparameters",
    ):
        if name not in handle:
            raise ValueError(f"{label}: missing '{name}' dataset")

    missing = [name for name in _REQUIRED_ATTRS if name not in handle.attrs]
    if missing:
        raise ValueError(f"{label}: missing attribute(s): {', '.join(missing)}")

    waveform_metadata_from_file(handle, label=label)
    frequency = np.asarray(handle["frequency"])
    spectra = np.asarray(handle["spectral_density"])
    n_events = np.asarray(handle["n_events"])
    rates = np.asarray(handle["total_merger_rate"])
    parameters = np.asarray(handle["hyperparameters"])

    if frequency.ndim != 1:
        raise ValueError(f"{label}: frequency dataset must be one-dimensional")
    if spectra.ndim != 2 or spectra.shape[1] != frequency.shape[0]:
        raise ValueError(
            f"{label}: 'spectral_density' must have shape (draws, frequency)"
        )
    draws = spectra.shape[0]
    if n_events.ndim != 1 or n_events.shape[0] != draws:
        raise ValueError(f"{label}: 'n_events' must have shape (draws,)")
    if rates.ndim != 1 or rates.shape[0] != draws:
        raise ValueError(f"{label}: 'total_merger_rate' must have shape (draws,)")
    if parameters.ndim != 2 or parameters.shape[0] != draws:
        raise ValueError(
            f"{label}: 'hyperparameters' must have shape (draws, parameter)"
        )
    names = _json_list(
        handle.attrs[PARAMETER_NAMES_ATTR], label=label, name=PARAMETER_NAMES_ATTR
    )
    if len(names) != parameters.shape[1] or len(set(names)) != len(names):
        raise ValueError(
            f"{label}: hyperparameter ordering does not match hyperparameters dataset"
        )
    _average_mode(str(_scalar(handle.attrs[AVERAGE_MODE_ATTR], name=AVERAGE_MODE_ATTR)))


def waveform_metadata_from_file(
    handle: h5py.Group, *, label: str
) -> PolarizationPowerGenerator:
    missing = [name for name in WAVEFORM_ATTRS if name not in handle.attrs]
    if missing:
        raise ValueError(
            f"{label}: missing waveform metadata attribute(s): {', '.join(missing)}"
        )
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
            frequency_resolution=float(
                _scalar(
                    handle.attrs["frequency_resolution"],
                    name="frequency_resolution",
                )
            ),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: invalid waveform metadata: {error}") from error


def _average_mode(value: str) -> AverageMode:
    for mode in AVERAGE_MODES:
        if value == mode:
            return mode
    raise ValueError(f"average_mode is {value!r}, expected one of {AVERAGE_MODES}")


def _check_format(attrs: Mapping[Any, Any], *, label: str) -> None:
    if attrs.get("format_name") != FORMAT_NAME:
        raise ValueError(
            f"{label}: format_name is {attrs.get('format_name')!r}, expected {FORMAT_NAME!r}"
        )
    if attrs.get("domain") != DOMAIN_FREQUENCY:
        raise ValueError(
            f"{label}: domain is {attrs.get('domain')!r}, expected {DOMAIN_FREQUENCY!r}"
        )


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
