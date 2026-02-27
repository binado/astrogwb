# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gwfast @ git+https://github.com/CosmoStatGW/gwfast.git",
#   "numpy<1.27",
# ]
# ///
"""Generate GWFast ORF reference fixtures.

Run once in a throwaway environment via::

    uv run --script scripts/generate_orf_fixtures.py

Saves ``tests/fixtures/gwfast_orf_reference.npz`` with arrays:
``frequencies``, ``H1_L1``, ``H1_V1``, ``L1_V1`` (50 points, 20–2048 Hz geomspace grid).
"""

from __future__ import annotations

import pathlib

import numpy as np
from gwfast.gwfastGlobals import detectors
from gwfast.stochastic import stochasticTools as st

PAIRS = [("H1", "L1"), ("H1", "V1"), ("L1", "V1")]
DETECTOR_NAME_MAP = {
    "H1": "H1",
    "L1": "L1",
    "V1": "Virgo",
}
FREQUENCIES = np.geomspace(20, 2048, 50)
OUT_PATH = (
    pathlib.Path(__file__).parent.parent
    / "tests"
    / "fixtures"
    / "gwfast_orf_reference.npz"
)


def gwfast_orf(det1_name: str, det2_name: str, freqs: np.ndarray) -> np.ndarray:
    det1_name_mapped = DETECTOR_NAME_MAP[det1_name]
    det2_name_mapped = DETECTOR_NAME_MAP[det2_name]
    det_1 = detectors[det1_name_mapped]
    det_2 = detectors[det2_name_mapped]
    result = st.overlap_reduction_function(
        freqs, det_1, det_2, det1_name=det1_name, det2_name=det2_name
    )
    return result[f"{det1_name}-{det2_name}"]


def main() -> None:
    data: dict[str, np.ndarray] = {"frequencies": FREQUENCIES}
    for det1, det2 in PAIRS:
        key = f"{det1}_{det2}"
        print(f"Computing ORF for {det1}-{det2}...")
        data[key] = gwfast_orf(det1, det2, FREQUENCIES)
        # print(f"  {key}: {data[key][:3]}...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **data)
    print(f"\nSaved fixture to {OUT_PATH}")


if __name__ == "__main__":
    main()
