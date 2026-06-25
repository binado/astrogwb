"""Numeric regression gate for the gwmock-native detector refactor.

These fixtures were snapshotted from the pre-refactor astrogwb code
(``Detector``/``PowerSpectralDensity``) and must stay byte-for-byte
reproducible through every phase of the refactor. They lock:

* the analytic frequency-dependent ORF (angle-based ``g1/g2/g3`` physics),
* the inverse-variance ``effective_psd`` contraction and the
  ``_NETWORK_SENSITIVITY_FACTOR = 0.16`` factor,
* the ``out_of_band="inf"`` PSD semantics that ``evaluate_psd`` must
  reproduce (out-of-band frequencies map to ``inf``, endpoints stay finite).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwmock_signal.network import Network

from astrogwb.detector import (
    effective_psd,
    evaluate_psd,
    load_sensitivities_for_network,
    overlap_reduction_function,
    pairwise_overlap_reduction_function,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, np.ndarray]:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"Missing regression fixture: {path.name}.")
    return dict(np.load(path))


def test_regression_orf() -> None:
    fixture = _load("regression_orf.npz")
    freqs = fixture["frequencies"]

    np.testing.assert_allclose(
        overlap_reduction_function(freqs, "H1", "L1"), fixture["H1_L1"]
    )
    np.testing.assert_allclose(
        overlap_reduction_function(freqs, "H1", "V1"), fixture["H1_V1"]
    )

    pairwise = pairwise_overlap_reduction_function(freqs, ["E1", "E2", "E3"])
    et_sum = pairwise[0, 1, :] + pairwise[0, 2, :] + pairwise[1, 2, :]
    np.testing.assert_allclose(et_sum, fixture["et_triangle_sum"])


def test_regression_effective_psd() -> None:
    fixture = _load("regression_effective_psd.npz")
    freqs = fixture["frequencies"]

    for network_name, key in (("H1L1", "H1_L1"), ("H1L1V1", "H1_L1_V1")):
        network = Network.from_name(network_name)
        actual = effective_psd(
            freqs,
            network.detector_names,
            load_sensitivities_for_network(network),
        )
        np.testing.assert_allclose(actual, fixture[key])


def test_regression_psd_evaluate() -> None:
    fixture = _load("regression_psd_evaluate.npz")
    values = evaluate_psd("AplusDesign_psd.txt", fixture["frequencies"])
    np.testing.assert_allclose(values, fixture["values"])
