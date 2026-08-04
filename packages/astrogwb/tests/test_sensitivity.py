from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from astrogwb.detector import (
    Sensitivity,
    effective_psd,
    evaluate_psd,
    load_sensitivities_for_network,
    load_sensitivity,
    load_sensitivity_map,
    overlap_reduction_function,
)
from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network


def test_evaluate_psd_in_band_is_finite_and_positive() -> None:
    values = evaluate_psd("AplusDesign_psd.txt", np.array([25.0, 100.0, 500.0]))

    assert values.shape == (3,)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


@pytest.mark.parametrize(
    ("out_of_band", "expected_oob_value"),
    [("inf", np.inf), ("zero", 0.0)],
)
def test_evaluate_psd_out_of_band_policy(
    out_of_band: Literal["inf", "zero"], expected_oob_value: float
) -> None:
    # 1.0 Hz is below the AplusDesign grid (min 5 Hz); 9000 Hz is above (max 5000).
    values = evaluate_psd(
        "AplusDesign_psd.txt",
        np.array([1.0, 100.0, 9000.0]),
        out_of_band=out_of_band,
    )

    assert values[0] == expected_oob_value
    assert np.isfinite(values[1]) and values[1] > 0.0
    assert values[2] == expected_oob_value


def test_evaluate_psd_bundled_preset_runs() -> None:
    values = evaluate_psd("ET_D_psd", np.array([20.0, 100.0]))

    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_evaluate_psd_unknown_reference_raises() -> None:
    with pytest.raises(FileNotFoundError):
        evaluate_psd("does_not_exist_anywhere.txt", np.array([100.0]))


def test_sensitivity_evaluate_delegates() -> None:
    sensitivity = Sensitivity(psd_reference="AplusDesign_psd.txt")

    direct = evaluate_psd("AplusDesign_psd.txt", np.array([100.0]))
    np.testing.assert_allclose(sensitivity.evaluate(np.array([100.0])), direct)


def test_load_sensitivity_single() -> None:
    sensitivity = load_sensitivity("H1")

    values = sensitivity.evaluate(np.array([100.0]))
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_load_sensitivity_map_multiple() -> None:
    sensitivities = load_sensitivity_map(["H1", "L1", "V1"])

    assert set(sensitivities) == {"H1", "L1", "V1"}
    for sensitivity in sensitivities.values():
        values = sensitivity.evaluate(np.array([100.0]))
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


@pytest.mark.integration
def test_load_sensitivities_for_network_et_preset(frequencies: np.ndarray) -> None:
    network = Network.from_name("ET-Triangle-Sardinia")
    sensitivities = load_sensitivities_for_network(network)

    names = network.detector_names
    assert all(isinstance(d, CustomDetector) for d in names)
    assert set(sensitivities) == {"ET1_SARD", "ET2_SARD", "ET3_SARD"}

    eff = effective_psd(frequencies, network.detector_names, sensitivities)
    assert np.any(np.isfinite(eff))
