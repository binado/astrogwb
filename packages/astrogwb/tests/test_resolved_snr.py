from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from astrogwb.detector import Sensitivity, load_detector, load_sensitivity_map
from astrogwb.resolved import REQUIRED_SOURCE_PARAMETERS, optimal_snr
from astrogwb.resolved import snr as snr_module
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


def _event(**overrides: float) -> dict[str, np.ndarray]:
    parameters = {key: value.copy() for key, value in BASE_PARAMETERS.items()}
    for key, value in overrides.items():
        parameters[key] = np.array([value])
    return parameters


def _stack_events(*events: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = events[0].keys()
    return {key: np.concatenate([event[key] for event in events]) for key in keys}


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


def test_required_source_parameters_are_exported() -> None:
    assert REQUIRED_SOURCE_PARAMETERS == snr_module.REQUIRED_SOURCE_PARAMETERS
    for key in REQUIRED_SOURCE_PARAMETERS:
        assert key in BASE_PARAMETERS


def test_invalid_backend_rejected() -> None:
    invalid_backend: Any = "numpy"
    with pytest.raises(ValueError, match="backend"):
        optimal_snr(
            BASE_PARAMETERS,
            *_h1_setup(),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            backend=invalid_backend,
        )


def test_normalize_parameters_drops_catalog_metadata() -> None:
    extras = {
        **BASE_PARAMETERS,
        "redshift": np.array([0.1]),
        "source_frame_mass_1": np.array([1.2]),
        "inclination": np.array([0.3]),
        "lambda_1": np.array([400.0]),
    }

    event_arrays, n_events = snr_module._normalize_parameters(extras)

    assert n_events == 1
    assert "redshift" not in event_arrays
    assert "source_frame_mass_1" not in event_arrays
    np.testing.assert_array_equal(event_arrays["inclination"], [0.3])
    np.testing.assert_array_equal(event_arrays["lambda_1"], [400.0])
    for key in REQUIRED_SOURCE_PARAMETERS:
        np.testing.assert_array_equal(event_arrays[key], BASE_PARAMETERS[key])


def test_normalize_parameters_ignores_malformed_catalog_metadata() -> None:
    extras = {
        **BASE_PARAMETERS,
        "redshift": np.ones((2, 2)),
        "source_frame_mass_1": 0.0,
    }

    event_arrays, n_events = snr_module._normalize_parameters(extras)

    assert n_events == 1
    assert set(event_arrays) == set(REQUIRED_SOURCE_PARAMETERS)


def test_matched_filter_snr_sinusoid_in_white_noise() -> None:
    sampling_frequency = 256.0
    n_samples = 256
    duration = n_samples / sampling_frequency
    dt = 1.0 / sampling_frequency
    time = np.arange(n_samples) * dt
    amplitude = 2.0
    psd_level = 4.0
    strain = (amplitude * np.cos(2.0 * np.pi * 32.0 * time)).reshape(1, 1, -1)
    frequencies, delta_f, grid_dt = snr_module._rfft_grid(n_samples, sampling_frequency)
    assert grid_dt == dt
    mask = np.ones(frequencies.size, dtype=bool)
    inv_psd = np.full((1, frequencies.size), 1.0 / psd_level)
    snr = snr_module.matched_filter_snr(strain, inv_psd, mask, delta_f, dt)
    expected = amplitude * np.sqrt(duration / psd_level)
    np.testing.assert_allclose(snr, [[expected]], rtol=0.02)


@pytest.mark.integration
def test_ripple_backend_rejects_unsupported_approximant() -> None:
    with pytest.raises(ValueError, match="Ripple backend is not available"):
        optimal_snr(
            BASE_PARAMETERS,
            *_h1_setup(),
            waveform_model="SpinTaylorT4",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            backend="ripple",
        )


@pytest.mark.integration
def test_ripple_and_lal_paths_agree() -> None:
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
    }

    ripple_snrs = optimal_snr(BASE_PARAMETERS, *_h1_setup(), **kwargs)
    lal_snrs = optimal_snr(BASE_PARAMETERS, *_h1_setup(), backend="lal", **kwargs)

    assert lal_snrs.shape == ripple_snrs.shape
    np.testing.assert_allclose(lal_snrs, ripple_snrs, rtol=0.10)


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


@pytest.mark.integration
def test_lal_path_ignores_catalog_metadata() -> None:
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
        "backend": "lal",
    }
    extras = {
        **BASE_PARAMETERS,
        "redshift": np.array([0.5]),
        "source_frame_mass_1": np.array([1.1]),
    }

    baseline = optimal_snr(BASE_PARAMETERS, *_h1_setup(), **kwargs)
    with_extras = optimal_snr(extras, *_h1_setup(), **kwargs)

    np.testing.assert_allclose(with_extras, baseline)


@pytest.mark.integration
def test_lal_mixed_mass_catalog_matches_single_event_snrs() -> None:
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
        "backend": "lal",
    }
    light = _event(detector_frame_mass_1=1.0, detector_frame_mass_2=1.0)
    heavy = _event(detector_frame_mass_1=2.5, detector_frame_mass_2=2.5)
    mixed = _stack_events(light, heavy)

    light_snr = optimal_snr(light, *_h1_setup(), **kwargs)
    heavy_snr = optimal_snr(heavy, *_h1_setup(), **kwargs)
    mixed_snr = optimal_snr(mixed, *_h1_setup(), **kwargs)

    assert mixed_snr.shape == (2, 1)
    np.testing.assert_allclose(mixed_snr[0], light_snr[0], rtol=0.02)
    np.testing.assert_allclose(mixed_snr[1], heavy_snr[0], rtol=0.02)
