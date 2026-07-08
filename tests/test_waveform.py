from __future__ import annotations

import numpy as np
from waveform_catalog import WaveformCatalog

from astrogwb.waveform import polarization_power


def test_polarization_power_reduces_catalog() -> None:
    catalog = WaveformCatalog(
        frequencies=np.array([10.0, 20.0, 30.0]),
        plus=np.array([[1.0 + 1.0j, 4.0], [2.0, 5.0 + 1.0j], [3.0, 6.0]]),
        cross=np.array([[0.0, 1.0j], [1.0, 0.0], [0.0, 2.0]]),
        source_parameters={"mass_1": np.array([20.0, 30.0])},
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=30.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
    )

    actual = polarization_power(catalog)

    expected = np.array(
        [
            [2.0, 17.0],
            [5.0, 26.0],
            [9.0, 40.0],
        ]
    )
    assert actual.shape == (3, 2)  # (nfreq, nsamples)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected)
