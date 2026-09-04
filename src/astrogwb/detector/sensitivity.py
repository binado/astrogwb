"""Detector sensitivity curves backed by gwmock-noise.

gwmock-noise owns PSD loading/interpolation; astrogwb only adds the SGWB
analysis policy on top. The one numerical wrinkle is out-of-band behavior:
``gwmock_noise`` clips frequencies outside the curve grid to ``0`` (an
*infinite* sensitivity contribution in an inverse-variance sum), whereas
the SGWB analysis wants those bins to contribute *nothing*, i.e. PSD
``inf``. ``evaluate_psd`` therefore re-applies the ``inf`` policy on top of
the gwmock interpolation.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from gwmock_noise.gaussian.psd import (
    is_remote_psd_reference,
    resolve_bundled_psd_preset,
)
from gwmock_noise.spectral import interpolate_real_spectral_series, load_spectral_series
from gwmock_signal.network import Network
from gwmock_signal.stochastic.overlap import detector_names
from numpy.typing import ArrayLike, NDArray

NOISE_CURVES_BASE_DIR = Path(__file__).parent / "noise_curves"
SENSITIVITY_FILE = Path(__file__).parent / "sensitivity.toml"

OutOfBand = Literal["inf", "zero"]


@dataclass(frozen=True, slots=True)
class Sensitivity:
    """Noise curve reference for one detector.

    ``psd_reference`` may be a gwmock-noise bundled preset name (e.g.
    ``"ET_D_psd"``), a file in astrogwb's ``noise_curves/`` directory, an
    absolute path, or an HTTP(S) URL.

    ``sensitivity.toml`` may also list ``minimum_frequency``,
    ``maximum_frequency``, and ``duty_factor`` as reference metadata for
    analysis setup; those fields are not loaded into this object.
    """

    psd_reference: str | Path

    def evaluate(
        self, frequencies: ArrayLike, *, out_of_band: OutOfBand = "inf"
    ) -> NDArray[np.float64]:
        return evaluate_psd(self.psd_reference, frequencies, out_of_band=out_of_band)


def resolve_psd_path(reference: str | Path) -> Path | str:
    """Resolve a PSD reference to something gwmock-noise can load.

    Resolution order: gwmock-noise bundled preset -> astrogwb
    ``noise_curves/`` file -> absolute/relative path on disk -> HTTP(S) URL.
    Bundled presets and on-disk files return a ``Path``; URLs return the
    original ``str`` (gwmock-noise fetches them directly).
    """
    reference_str = str(reference)

    bundled = resolve_bundled_psd_preset(reference_str)
    if bundled is not None:
        return bundled

    packaged = NOISE_CURVES_BASE_DIR / reference_str
    if packaged.exists():
        return packaged

    path = Path(reference)
    if path.exists():
        return path

    if is_remote_psd_reference(reference_str):
        return reference_str

    raise FileNotFoundError(f"Could not resolve PSD reference: {reference!r}")


def evaluate_psd(
    reference: str | Path,
    frequencies: ArrayLike,
    *,
    out_of_band: OutOfBand = "inf",
) -> NDArray[np.float64]:
    """Interpolate a PSD onto ``frequencies`` using gwmock-noise.

    ``out_of_band="zero"`` returns gwmock-noise's raw interpolation
    (frequencies outside the curve grid clipped to ``0``).
    ``out_of_band="inf"`` (default) instead maps those frequencies to
    ``inf``, the astrogwb analysis convention so out-of-band bins drop out
    of an inverse-variance contraction. Grid endpoints stay finite.
    """
    resolved = resolve_psd_path(reference)
    frequencies = np.asarray(frequencies, dtype=float)
    grid, grid_values = load_spectral_series(resolved, kind="PSD")
    values = interpolate_real_spectral_series(grid, grid_values, frequencies)

    if out_of_band == "zero":
        return values

    out_of_band_mask = (frequencies < grid.min()) | (frequencies > grid.max())
    return np.where(out_of_band_mask, np.inf, values)


def load_sensitivity(name: str, *, path: str | Path | None = None) -> Sensitivity:
    """Load a single detector's :class:`Sensitivity` from a TOML table."""
    table = _load_sensitivity_table(path)
    return _sensitivity_from_dict(table[name])


def load_sensitivity_map(
    names: Sequence[str], *, path: str | Path | None = None
) -> Mapping[str, Sensitivity]:
    """Load a name -> :class:`Sensitivity` mapping for several detectors."""
    table = _load_sensitivity_table(path)
    return {name: _sensitivity_from_dict(table[name]) for name in names}


def load_sensitivities_for_network(
    network: Network | str, *, path: str | Path | None = None
) -> Mapping[str, Sensitivity]:
    """Load sensitivity curves keyed by each detector's public name.

    ``network`` may be a gwmock preset alias (see ``Network.list_names()``)
    or a ready :class:`~gwmock_signal.network.Network`.
    """
    if isinstance(network, str):
        network = Network.from_name(network)
    return load_sensitivity_map(detector_names(network.detector_names), path=path)


def _load_sensitivity_table(path: str | Path | None) -> Mapping[str, dict]:
    path = Path(path) if path is not None else SENSITIVITY_FILE
    with open(path, "rb") as f:
        return tomllib.load(f)


def _sensitivity_from_dict(data: Mapping) -> Sensitivity:
    return Sensitivity(psd_reference=data["psd_reference"])
