from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from astrogwb.detector import Sensitivity, load_detector, load_sensitivity_map
from astrogwb.resolved import optimal_snr
from gwmock_signal.detector import CustomDetector

BASE_PARAMETERS: dict[str, np.ndarray] = {
    "coa_time": np.array([0.0]),
    "right_ascension": np.array([0.0]),
    "declination": np.array([0.0]),
    "polarization_angle": np.array([0.0]),
    "detector_frame_mass_1": np.array([1.4]),
    "detector_frame_mass_2": np.array([1.4]),
    "luminosity_distance": np.array([100.0]),
}


def _h1_setup() -> tuple[list[CustomDetector], Mapping[str, Sensitivity]]:
    return [load_detector("H1")], load_sensitivity_map(["H1"])


def test_missing_required_parameter_is_reported() -> None:
    incomplete = {
        key: value for key, value in BASE_PARAMETERS.items() if key != "coa_time"
    }

    with pytest.raises(ValueError, match="coa_time"):
        optimal_snr(
            incomplete,
            *_h1_setup(),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


def test_missing_required_parameters_are_reported_together() -> None:
    incomplete = {
        key: value
        for key, value in BASE_PARAMETERS.items()
        if key
        not in {"coa_time", "right_ascension", "declination", "polarization_angle"}
    }

    with pytest.raises(ValueError) as excinfo:
        optimal_snr(
            incomplete,
            *_h1_setup(),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )

    for key in (
        "coa_time",
        "right_ascension",
        "declination",
        "polarization_angle",
    ):
        assert key in str(excinfo.value)


def test_empty_source_parameters_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        optimal_snr(
            {},
            *_h1_setup(),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


def test_empty_parameter_arrays_rejected() -> None:
    parameters = {key: value[:0] for key, value in BASE_PARAMETERS.items()}

    with pytest.raises(ValueError, match="non-empty"):
        optimal_snr(
            parameters,
            *_h1_setup(),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


def test_scalar_parameter_rejected() -> None:
    parameters = {**BASE_PARAMETERS, "inclination": 0.0}

    with pytest.raises(ValueError, match="1-dimensional"):
        optimal_snr(
            parameters,
            *_h1_setup(),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


def test_mismatched_parameter_lengths_rejected() -> None:
    parameters = {
        **{key: np.full(2, value[0]) for key, value in BASE_PARAMETERS.items()},
        "lambda_1": np.array([100.0, 200.0]),
        "lambda_2": np.array([100.0, 200.0, 300.0]),
    }

    with pytest.raises(ValueError, match="lambda_2"):
        optimal_snr(
            parameters,
            *_h1_setup(),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


def test_multidimensional_parameter_rejected() -> None:
    parameters = {
        **BASE_PARAMETERS,
        "detector_frame_mass_1": np.ones((2, 2)),
    }

    with pytest.raises(ValueError, match="1-dimensional"):
        optimal_snr(
            parameters,
            *_h1_setup(),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sampling_frequency", 0.0, "sampling_frequency"),
        ("sampling_frequency", -512.0, "sampling_frequency"),
        ("minimum_frequency", 0.0, "minimum_frequency"),
        ("minimum_frequency", -20.0, "minimum_frequency"),
    ],
)
def test_invalid_scalar_arguments_rejected(
    field: str, value: float, match: str
) -> None:
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
        field: value,
    }

    with pytest.raises(ValueError, match=match):
        optimal_snr(BASE_PARAMETERS, *_h1_setup(), **kwargs)


def test_maximum_frequency_below_minimum_rejected() -> None:
    with pytest.raises(ValueError, match="maximum_frequency"):
        optimal_snr(
            BASE_PARAMETERS,
            *_h1_setup(),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            maximum_frequency=10.0,
        )


def test_empty_detector_list_rejected() -> None:
    with pytest.raises(ValueError, match="At least one detector"):
        optimal_snr(
            BASE_PARAMETERS,
            [],
            load_sensitivity_map(["H1"]),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


def test_duplicate_detector_names_rejected() -> None:
    def custom(name: str) -> CustomDetector:
        return CustomDetector(
            name=name,
            latitude_rad=0.0,
            longitude_rad=0.0,
            elevation_m=0.0,
            xarm_azimuth_rad=0.0,
            yarm_azimuth_rad=np.pi / 2,
        )

    with pytest.raises(ValueError, match="Duplicate detector name"):
        optimal_snr(
            BASE_PARAMETERS,
            [custom("XX"), custom("XX")],
            {"XX": load_sensitivity_map(["H1"])["H1"]},
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


def test_missing_sensitivity_is_reported() -> None:
    with pytest.raises(KeyError, match="V1"):
        optimal_snr(
            BASE_PARAMETERS,
            [load_detector("V1")],
            load_sensitivity_map(["H1"]),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


@pytest.mark.integration
def test_progress_callback_reports_completion() -> None:
    calls: list[tuple[int, int]] = []

    snrs = optimal_snr(
        BASE_PARAMETERS,
        *_h1_setup(),
        waveform_model="IMRPhenomXAS_NRTidalv3",
        sampling_frequency=512.0,
        minimum_frequency=20.0,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert snrs.shape == (1, 1)
    assert calls[-1] == (1, 1)


@pytest.mark.integration
def test_two_detector_snr_aligns_with_detector_order() -> None:
    detectors_hv = [load_detector("H1"), load_detector("V1")]
    detectors_vh = [load_detector("V1"), load_detector("H1")]
    sensitivities = load_sensitivity_map(["H1", "V1"])
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
    }

    hv = optimal_snr(BASE_PARAMETERS, detectors_hv, sensitivities, **kwargs)
    vh = optimal_snr(BASE_PARAMETERS, detectors_vh, sensitivities, **kwargs)

    assert hv.shape == (1, 2)
    assert np.all(np.isfinite(hv))
    assert np.all(hv >= 0.0)
    assert np.allclose(hv, vh[:, ::-1])
