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
from gwmock_noise.spectral import interpolate_real_spectral_series, load_spectral_series
from gwmock_signal.network import Network
from gwmock_signal.stochastic.overlap import detector_names
from numpy.typing import ArrayLike, NDArray

from astrogwb.psd import NOISE_CURVES_BASE_DIR as NOISE_CURVES_BASE_DIR  # noqa: PLC0414
from astrogwb.psd import resolve_psd_path

SENSITIVITY_FILE = Path(__file__).parent / "sensitivity.toml"

OutOfBand = Literal["inf", "zero"]


@dataclass(frozen=True, slots=True)
class Sensitivity:
    """Noise curve reference for one detector.

    ``psd_reference`` may be a gwmock-noise bundled preset name (e.g.
    ``"ET_D_psd"``), a file in astrogwb's ``noise_curves/`` directory, an
    absolute path, or an HTTP(S) URL.
    """

    psd_reference: str | Path

    def evaluate(
        self, frequencies: ArrayLike, *, out_of_band: OutOfBand = "inf"
    ) -> NDArray[np.float64]:
        return evaluate_psd(self.psd_reference, frequencies, out_of_band=out_of_band)


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
    if path is not None:
        with Path(path).open("rb") as handle:
            data = tomllib.load(handle)
        # Keep accepting the flat layout used by existing external tables.
        return data.get("detectors", data)
    with SENSITIVITY_FILE.open("rb") as handle:
        return tomllib.load(handle)["detectors"]


def _sensitivity_from_dict(data: Mapping) -> Sensitivity:
    return Sensitivity(psd_reference=data["psd_reference"])
