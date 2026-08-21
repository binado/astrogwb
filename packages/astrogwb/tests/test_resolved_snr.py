from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from astrogwb.resolved import network_optimal_snr
from gwmock_signal.detector import CustomDetector

BASE_PARAMETERS: dict[str, float] = {
    "tc": 0.0,
    "ra": 0.0,
    "dec": 0.0,
    "psi": 0.0,
    "detector_frame_mass_1": 1.4,
    "detector_frame_mass_2": 1.4,
    "luminosity_distance": 100.0,
}


def test_missing_required_parameter_is_reported() -> None:
    incomplete = {key: value for key, value in BASE_PARAMETERS.items() if key != "tc"}

    with pytest.raises(ValueError, match="tc"):
        network_optimal_snr(
            incomplete,
            ["H1"],
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


def test_missing_required_parameters_are_reported_together() -> None:
    incomplete = {
        key: value
        for key, value in BASE_PARAMETERS.items()
        if key not in {"tc", "ra", "dec", "psi"}
    }

    with pytest.raises(ValueError) as excinfo:
        network_optimal_snr(
            incomplete,
            ["H1"],
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )

    for key in ("tc", "ra", "dec", "psi"):
        assert key in str(excinfo.value)


def test_empty_source_parameters_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        network_optimal_snr(
            {},
            ["H1"],
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )


def test_mismatched_parameter_lengths_rejected() -> None:
    parameters = {
        **BASE_PARAMETERS,
        "lambda_1": np.array([100.0, 200.0]),
        "lambda_2": np.array([100.0, 200.0, 300.0]),
    }

    with pytest.raises(ValueError, match="lambda_2"):
        network_optimal_snr(
            parameters,
            ["H1"],
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
        network_optimal_snr(
            parameters,
            ["H1"],
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
        ("chunk_size", -1, "chunk_size"),
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
        network_optimal_snr(BASE_PARAMETERS, ["H1"], **kwargs)


def test_maximum_frequency_below_minimum_rejected() -> None:
    with pytest.raises(ValueError, match="maximum_frequency"):
        network_optimal_snr(
            BASE_PARAMETERS,
            ["H1"],
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            maximum_frequency=10.0,
        )


def test_empty_detector_list_rejected() -> None:
    with pytest.raises(ValueError, match="At least one detector"):
        network_optimal_snr(
            BASE_PARAMETERS,
            [],
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
        network_optimal_snr(
            BASE_PARAMETERS,
            [custom("XX"), custom("XX")],
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
        )
