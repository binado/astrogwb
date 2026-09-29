from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from gwmock_signal.network import Network

from astrogwb.detector import (
    Sensitivity,
    effective_psd,
    load_sensitivities_for_network,
    load_sensitivity_map,
    overlap_reduction_function,
)


def test_sensitivity_call_in_band_is_finite_and_positive() -> None:
    sensitivity = Sensitivity(psd_reference="AplusDesign_psd.txt")
    values = sensitivity(np.array([25.0, 100.0, 500.0]))

    assert values.shape == (3,)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


@pytest.mark.parametrize(
    ("out_of_band", "expected_oob_value"),
    [("inf", np.inf), ("zero", 0.0)],
)
def test_sensitivity_call_out_of_band_policy(
    out_of_band: Literal["inf", "zero"], expected_oob_value: float
) -> None:
    # 1.0 Hz is below the AplusDesign grid (min 5 Hz); 9000 Hz is above (max 5000).
    sensitivity = Sensitivity(psd_reference="AplusDesign_psd.txt")
    values = sensitivity(np.array([1.0, 100.0, 9000.0]), out_of_band=out_of_band)

    assert values[0] == expected_oob_value
    assert np.isfinite(values[1]) and values[1] > 0.0
    assert values[2] == expected_oob_value


def test_sensitivity_call_bundled_preset_runs() -> None:
    sensitivity = Sensitivity(psd_reference="ET_D_psd")
    values = sensitivity(np.array([20.0, 100.0]))

    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_sensitivity_call_unknown_reference_raises() -> None:
    sensitivity = Sensitivity(psd_reference="does_not_exist_anywhere.txt")
    with pytest.raises(FileNotFoundError):
        sensitivity(np.array([100.0]))


def test_load_sensitivity_map_multiple() -> None:
    sensitivities = load_sensitivity_map(["H1", "L1", "V1"])

    assert set(sensitivities) == {"H1", "L1", "V1"}
    for sensitivity in sensitivities.values():
        values = sensitivity(np.array([100.0]))
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)


def test_load_sensitivities_for_network_str_network() -> None:
    sensitivities = load_sensitivities_for_network("HLVK")

    assert set(sensitivities) == {"H1", "L1", "V1", "K1"}


def test_network_overlap_and_effective_psd(frequencies: np.ndarray) -> None:
    network = Network.from_name("H1L1V1")
    sensitivities = load_sensitivities_for_network(network)

    orf = overlap_reduction_function(frequencies, "H1", "L1")
    assert orf.shape == frequencies.shape
    assert np.all(np.isfinite(orf))

    eff = effective_psd(frequencies, network.detector_names, sensitivities)
    assert eff.shape == frequencies.shape
    assert np.any(np.isfinite(eff))


def test_effective_psd_inf_for_single_detector(frequencies: np.ndarray) -> None:
    actual = effective_psd(frequencies, ["H1"], load_sensitivity_map(["H1"]))

    assert actual.shape == frequencies.shape
    assert np.all(np.isinf(actual))
