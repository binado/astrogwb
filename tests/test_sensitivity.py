from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gwmock_signal.detector import CustomDetector

from astrogwb.detector import (
    Sensitivity,
    effective_psd,
    evaluate_psd,
    load_sensitivity,
    load_sensitivity_map,
    load_sensitivities_for_network,
    overlap_reduction_function,
)
from astrogwb.detector.sensitivity import resolve_psd_path


@pytest.fixture
def frequencies() -> np.ndarray:
    return np.geomspace(20, 2048, 128)


def test_resolve_psd_path_bundled_preset() -> None:
    path = resolve_psd_path("ET_D_psd")

    assert isinstance(path, Path)
    assert path.name == "ET_D_psd.txt"
    assert path.exists()


def test_resolve_psd_path_local_noise_curve() -> None:
    path = resolve_psd_path("AplusDesign_psd.txt")

    assert isinstance(path, Path)
    assert path.name == "AplusDesign_psd.txt"
    assert path.parent.name == "noise_curves"


def test_resolve_psd_path_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_psd_path("does_not_exist_anywhere.txt")


def test_evaluate_psd_in_band_is_finite_and_positive() -> None:
    values = evaluate_psd("AplusDesign_psd.txt", np.array([25.0, 100.0, 500.0]))

    assert values.shape == (3,)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_evaluate_psd_out_of_band_inf_policy() -> None:
    # 1.0 Hz is below the AplusDesign grid (min 5 Hz); 9000 Hz is above (max 5000).
    values = evaluate_psd("AplusDesign_psd.txt", np.array([1.0, 100.0, 9000.0]))

    assert np.isinf(values[0])
    assert np.isfinite(values[1])
    assert np.isinf(values[2])


def test_evaluate_psd_out_of_band_zero_policy() -> None:
    values = evaluate_psd(
        "AplusDesign_psd.txt", np.array([1.0, 100.0, 9000.0]), out_of_band="zero"
    )

    assert values[0] == 0.0
    assert np.isfinite(values[1]) and values[1] > 0.0
    assert values[2] == 0.0


def test_evaluate_psd_bundled_preset_runs() -> None:
    values = evaluate_psd("ET_D_psd", np.array([20.0, 100.0]))

    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_sensitivity_evaluate_delegates() -> None:
    sensitivity = Sensitivity(psd_reference="AplusDesign_psd.txt")

    direct = evaluate_psd("AplusDesign_psd.txt", np.array([100.0]))
    np.testing.assert_allclose(sensitivity.evaluate(np.array([100.0])), direct)


def test_load_sensitivity_single() -> None:
    sensitivity = load_sensitivity("H1")

    assert sensitivity.psd_reference == "AplusDesign_psd.txt"


def test_load_sensitivity_map_multiple() -> None:
    sensitivities = load_sensitivity_map(["H1", "L1", "V1"])

    assert set(sensitivities) == {"H1", "L1", "V1"}
    assert sensitivities["V1"].psd_reference == "avirgo_O5low_NEW_psd.txt"


def test_load_sensitivities_for_network_str_network() -> None:
    sensitivities = load_sensitivities_for_network("HLVK")

    assert set(sensitivities) == {"H1", "L1", "V1", "K1"}


def test_network_overlap_and_effective_psd(frequencies: np.ndarray) -> None:
    from gwmock_signal.network import Network

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
    from gwmock_signal.network import Network

    network = Network.from_name("ET-Triangle-Sardinia")
    sensitivities = load_sensitivities_for_network(network)

    names = network.detector_names
    assert all(isinstance(d, CustomDetector) for d in names)
    assert set(sensitivities) == {"ET1_SARD", "ET2_SARD", "ET3_SARD"}

    eff = effective_psd(frequencies, network.detector_names, sensitivities)
    assert np.any(np.isfinite(eff))
