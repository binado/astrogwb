from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from astrogwb.detector import (
    Sensitivity,
    effective_psd,
    load_detector,
    load_sensitivity_map,
)


def test_bundled_sensitivities_are_finite_and_positive_in_band() -> None:
    sensitivities = [
        Sensitivity(psd_reference="AplusDesign_psd.txt"),
        Sensitivity(psd_reference="ET_D_psd"),
        *load_sensitivity_map(["H1", "L1", "V1"]).values(),
    ]
    frequencies = np.array([25.0, 100.0, 500.0])
    for sensitivity in sensitivities:
        values = sensitivity(frequencies)
        assert values.shape == frequencies.shape
        assert np.all(np.isfinite(values) & (values > 0.0))


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


def test_effective_psd_inf_for_single_detector(frequencies: np.ndarray) -> None:
    actual = effective_psd(frequencies, [load_detector("H1")])

    assert actual.shape == frequencies.shape
    assert np.all(np.isinf(actual))
