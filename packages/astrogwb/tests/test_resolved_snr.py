from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from astrogwb.detector import Sensitivity, load_detector, load_sensitivity_map
from astrogwb.resolved import optimal_snr
from astrogwb.resolved import snr as snr_module
from gwmock_signal.detector import CustomDetector

SourceParameters = dict[str, np.ndarray]
DetectorSetup = tuple[list[CustomDetector], Mapping[str, Sensitivity]]
ParameterFactory = Callable[..., SourceParameters]


@pytest.fixture
def base_parameters() -> SourceParameters:
    return {
        "coa_time": np.array([0.0]),
        "right_ascension": np.array([0.0]),
        "declination": np.array([0.0]),
        "polarization_angle": np.array([0.0]),
        "detector_frame_mass_1": np.array([1.4]),
        "detector_frame_mass_2": np.array([1.4]),
        "luminosity_distance": np.array([100.0]),
    }


@pytest.fixture
def h1_setup() -> DetectorSetup:
    return [load_detector("H1")], load_sensitivity_map(["H1"])


@pytest.fixture
def event_factory(
    base_parameters: SourceParameters,
) -> ParameterFactory:
    def make_event(**overrides: float) -> SourceParameters:
        parameters = {key: value.copy() for key, value in base_parameters.items()}
        for key, value in overrides.items():
            parameters[key] = np.array([value])
        return parameters

    return make_event


@pytest.fixture
def stack_events() -> ParameterFactory:
    def stack(*events: Mapping[str, np.ndarray]) -> SourceParameters:
        keys = events[0].keys()
        return {key: np.concatenate([event[key] for event in events]) for key in keys}

    return stack


def test_empty_source_parameters_rejected(
    h1_setup: DetectorSetup,
) -> None:
    with pytest.raises(ValueError, match="empty"):
        optimal_snr(
            {},
            *h1_setup,
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
        )


def test_empty_parameter_arrays_rejected(
    base_parameters: SourceParameters,
    h1_setup: DetectorSetup,
) -> None:
    parameters = {key: value[:0] for key, value in base_parameters.items()}

    with pytest.raises(ValueError, match="non-empty"):
        optimal_snr(
            parameters,
            *h1_setup,
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
        )


def test_scalar_parameter_rejected(
    base_parameters: SourceParameters,
    h1_setup: DetectorSetup,
) -> None:
    parameters = {**base_parameters, "inclination": 0.0}

    with pytest.raises(ValueError, match="1-dimensional"):
        optimal_snr(
            parameters,
            *h1_setup,
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
        )


def test_mismatched_parameter_lengths_rejected(
    base_parameters: SourceParameters,
    h1_setup: DetectorSetup,
) -> None:
    parameters = {
        **{key: np.full(2, value[0]) for key, value in base_parameters.items()},
        "lambda_1": np.array([100.0, 200.0]),
        "lambda_2": np.array([100.0, 200.0, 300.0]),
    }

    with pytest.raises(ValueError, match="lambda_2"):
        optimal_snr(
            parameters,
            *h1_setup,
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
        )


def test_multidimensional_parameter_rejected(
    base_parameters: SourceParameters,
    h1_setup: DetectorSetup,
) -> None:
    parameters = {
        **base_parameters,
        "detector_frame_mass_1": np.ones((2, 2)),
    }

    with pytest.raises(ValueError, match="1-dimensional"):
        optimal_snr(
            parameters,
            *h1_setup,
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
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
    field: str,
    value: float,
    match: str,
    base_parameters: SourceParameters,
    h1_setup: DetectorSetup,
) -> None:
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
        "batch_size": 1,
        field: value,
    }

    with pytest.raises(ValueError, match=match):
        optimal_snr(base_parameters, *h1_setup, **kwargs)


def test_maximum_frequency_below_minimum_rejected(
    base_parameters: SourceParameters, h1_setup: DetectorSetup
) -> None:
    with pytest.raises(ValueError, match="maximum_frequency"):
        optimal_snr(
            base_parameters,
            *h1_setup,
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
            maximum_frequency=10.0,
        )


def test_empty_detector_list_rejected(base_parameters: SourceParameters) -> None:
    with pytest.raises(ValueError, match="At least one detector"):
        optimal_snr(
            base_parameters,
            [],
            load_sensitivity_map(["H1"]),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
        )


def test_duplicate_detector_names_rejected(
    base_parameters: SourceParameters,
) -> None:
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
            base_parameters,
            [custom("XX"), custom("XX")],
            {"XX": load_sensitivity_map(["H1"])["H1"]},
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
        )


def test_missing_sensitivity_is_reported(
    base_parameters: SourceParameters,
) -> None:
    with pytest.raises(KeyError, match="V1"):
        optimal_snr(
            base_parameters,
            [load_detector("V1")],
            load_sensitivity_map(["H1"]),
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
        )


def test_invalid_backend_rejected(
    base_parameters: SourceParameters, h1_setup: DetectorSetup
) -> None:
    invalid_backend: Any = "numpy"
    with pytest.raises(ValueError, match="backend"):
        optimal_snr(
            base_parameters,
            *h1_setup,
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
            backend=invalid_backend,
        )


def test_non_positive_batch_size_rejected(
    base_parameters: SourceParameters, h1_setup: DetectorSetup
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        optimal_snr(
            base_parameters,
            *h1_setup,
            waveform_model="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=0,
        )


def test_duration_batches_are_stable_longest_first() -> None:
    durations = np.array([4.0, 8.0, 8.0, 2.0])

    batches = snr_module._duration_sorted_batches(durations, batch_size=2)

    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0][0], [1, 2])
    np.testing.assert_array_equal(batches[1][0], [0, 3])
    assert [duration for _, duration in batches] == [8.0, 4.0]


def test_normalize_parameters_drops_catalog_metadata(
    base_parameters: SourceParameters,
) -> None:
    extras = {
        **base_parameters,
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
    for key, values in base_parameters.items():
        np.testing.assert_array_equal(event_arrays[key], values)


def test_normalize_parameters_rejects_malformed_catalog_metadata(
    base_parameters: SourceParameters,
) -> None:
    extras = {
        **base_parameters,
        "redshift": np.ones((2, 2)),
        "source_frame_mass_1": 0.0,
    }

    with pytest.raises(ValueError, match="redshift"):
        snr_module._normalize_parameters(extras)


@pytest.mark.integration
def test_ripple_backend_rejects_unsupported_approximant(
    base_parameters: SourceParameters, h1_setup: DetectorSetup
) -> None:
    with pytest.raises(ValueError, match="Ripple backend is not available"):
        optimal_snr(
            base_parameters,
            *h1_setup,
            waveform_model="SpinTaylorT4",
            sampling_frequency=512.0,
            minimum_frequency=20.0,
            batch_size=1,
            backend="ripple",
        )


@pytest.mark.integration
def test_ripple_and_lal_paths_agree(
    base_parameters: SourceParameters, h1_setup: DetectorSetup
) -> None:
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
        "batch_size": 1,
    }

    ripple_snrs = optimal_snr(base_parameters, *h1_setup, **kwargs)
    lal_snrs = optimal_snr(base_parameters, *h1_setup, backend="lal", **kwargs)

    assert lal_snrs.shape == ripple_snrs.shape
    np.testing.assert_allclose(lal_snrs, ripple_snrs, rtol=0.10)


@pytest.mark.integration
def test_progress_callback_reports_completion(
    base_parameters: SourceParameters, h1_setup: DetectorSetup
) -> None:
    calls: list[tuple[int, int]] = []

    snrs = optimal_snr(
        base_parameters,
        *h1_setup,
        waveform_model="IMRPhenomXAS_NRTidalv3",
        sampling_frequency=512.0,
        minimum_frequency=20.0,
        batch_size=1,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert snrs.shape == (1, 1)
    assert calls[-1] == (1, 1)


@pytest.mark.integration
def test_two_detector_snr_aligns_with_detector_order(
    base_parameters: SourceParameters,
) -> None:
    detectors_hv = [load_detector("H1"), load_detector("V1")]
    detectors_vh = [load_detector("V1"), load_detector("H1")]
    sensitivities = load_sensitivity_map(["H1", "V1"])
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
        "batch_size": 1,
    }

    hv = optimal_snr(base_parameters, detectors_hv, sensitivities, **kwargs)
    vh = optimal_snr(base_parameters, detectors_vh, sensitivities, **kwargs)

    assert hv.shape == (1, 2)
    assert np.all(np.isfinite(hv))
    assert np.all(hv >= 0.0)
    assert np.allclose(hv, vh[:, ::-1])


@pytest.mark.integration
def test_lal_path_ignores_catalog_metadata(
    base_parameters: SourceParameters, h1_setup: DetectorSetup
) -> None:
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
        "batch_size": 1,
        "backend": "lal",
    }
    extras = {
        **base_parameters,
        "redshift": np.array([0.5]),
        "source_frame_mass_1": np.array([1.1]),
    }

    baseline = optimal_snr(base_parameters, *h1_setup, **kwargs)
    with_extras = optimal_snr(extras, *h1_setup, **kwargs)

    np.testing.assert_allclose(with_extras, baseline)


@pytest.mark.integration
def test_lal_mixed_mass_catalog_matches_single_event_snrs(
    event_factory: ParameterFactory,
    stack_events: ParameterFactory,
    h1_setup: DetectorSetup,
) -> None:
    kwargs: dict[str, Any] = {
        "waveform_model": "IMRPhenomXAS_NRTidalv3",
        "sampling_frequency": 512.0,
        "minimum_frequency": 20.0,
        "batch_size": 1,
        "backend": "lal",
    }
    light = event_factory(detector_frame_mass_1=1.0, detector_frame_mass_2=1.0)
    heavy = event_factory(detector_frame_mass_1=2.5, detector_frame_mass_2=2.5)
    mixed = stack_events(light, heavy)

    light_snr = optimal_snr(light, *h1_setup, **kwargs)
    heavy_snr = optimal_snr(heavy, *h1_setup, **kwargs)
    mixed_snr = optimal_snr(mixed, *h1_setup, **kwargs)

    assert mixed_snr.shape == (2, 1)
    np.testing.assert_allclose(mixed_snr[0], light_snr[0], rtol=0.02)
    np.testing.assert_allclose(mixed_snr[1], heavy_snr[0], rtol=0.02)
