from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from gwmock_signal.waveform import RippleBackend


class PolarizationPowerCatalog(NamedTuple):
    frequencies: ArrayLike
    polarization_power: ArrayLike
    samples: dict[str, ArrayLike]


class _BatchPolarizationBackend(Protocol):
    def generate_fd_polarizations_batch(
        self,
        approximant: str,
        *,
        sampling_frequency: float,
        minimum_frequency: float,
        parameters: dict[str, ArrayLike],
    ) -> Any: ...


def generate_catalog_polarization_power(
    samples: dict[str, ArrayLike],
    *,
    approximant: str,
    sampling_frequency: float,
    minimum_frequency: float,
    backend: _BatchPolarizationBackend | None = None,
) -> PolarizationPowerCatalog:
    waveform_backend: _BatchPolarizationBackend = (
        backend if backend is not None else RippleBackend()
    )
    polarizations = waveform_backend.generate_fd_polarizations_batch(
        approximant,
        sampling_frequency=sampling_frequency,
        minimum_frequency=minimum_frequency,
        parameters=samples,
    )
    polarization_power = (
        abs(polarizations.plus) ** 2 + abs(polarizations.cross) ** 2
    ).T
    return PolarizationPowerCatalog(
        frequencies=polarizations.frequencies,
        polarization_power=polarization_power,
        samples=dict(samples),
    )


def save_polarization_power_catalog(
    path: str | Path, catalog: PolarizationPowerCatalog
) -> None:
    frequencies = np.asarray(catalog.frequencies)
    polarization_power = np.asarray(catalog.polarization_power)
    _validate_catalog_shapes(frequencies, polarization_power, catalog.samples)

    payload = {
        "frequencies": frequencies,
        "polarization_power": polarization_power,
    }
    payload.update(
        {
            f"sample__{name}": np.asarray(values)
            for name, values in catalog.samples.items()
        }
    )
    _write_npz(path, payload)


def load_polarization_power_catalog(path: str | Path) -> PolarizationPowerCatalog:
    with np.load(path) as data:
        frequencies = np.asarray(data["frequencies"])
        polarization_power = np.asarray(data["polarization_power"])
        samples = {
            key.removeprefix("sample__"): np.asarray(data[key])
            for key in data.files
            if key.startswith("sample__")
        }

    _validate_catalog_shapes(frequencies, polarization_power, samples)
    return PolarizationPowerCatalog(
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
    )


def _validate_catalog_shapes(
    frequencies: NDArray[Any],
    polarization_power: NDArray[Any],
    samples: dict[str, ArrayLike],
) -> None:
    if frequencies.ndim != 1:
        raise ValueError("frequencies must be one-dimensional")
    if polarization_power.ndim != 2:
        raise ValueError("polarization_power must be two-dimensional")
    if polarization_power.shape[0] != frequencies.shape[0]:
        raise ValueError("polarization_power first axis must match frequencies")
    for name, values in samples.items():
        if np.asarray(values).shape[0] != polarization_power.shape[1]:
            raise ValueError(
                f"sample {name!r} length must match polarization_power sample axis"
            )


def _write_npz(path: str | Path, payload: dict[str, NDArray[Any]]) -> None:
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, values in payload.items():
            buffer = io.BytesIO()
            np.save(buffer, values)
            archive.writestr(f"{name}.npy", buffer.getvalue())
