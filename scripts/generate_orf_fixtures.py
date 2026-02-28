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
``frequencies``, ``H1_L1``, ``H1_V1``, ``L1_V1`` and
``tests/fixtures/gwfast_orf_reference_et_triangle.npz`` with
``frequencies``, ``sum_upper_pairs`` (50 points, 20–2048 Hz geomspace grid).
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
from gwfast.gwfastGlobals import detectors
from gwfast.stochastic import stochasticTools as st

HLV_PAIRS = [("H1", "L1"), ("H1", "V1"), ("L1", "V1")]
DETECTOR_NAME_MAP = {
    "H1": "H1",
    "L1": "L1",
    "V1": "Virgo",
}
FREQUENCIES = np.geomspace(20, 2048, 50)
ET_ARM_LENGTH_KM = 10.0
HLV_OUT_PATH = (
    pathlib.Path(__file__).parent.parent
    / "tests"
    / "fixtures"
    / "gwfast_orf_reference.npz"
)
ET_OUT_PATH = (
    pathlib.Path(__file__).parent.parent
    / "tests"
    / "fixtures"
    / "gwfast_orf_reference_et_triangle.npz"
)
_ETS_KEY_RE = re.compile(r"^ETS_(\d)-ETS_(\d)$")


def gwfast_orf(det1_name: str, det2_name: str, freqs: np.ndarray) -> np.ndarray:
    det1_name_mapped = DETECTOR_NAME_MAP[det1_name]
    det2_name_mapped = DETECTOR_NAME_MAP[det2_name]
    det_1 = detectors[det1_name_mapped]
    det_2 = detectors[det2_name_mapped]
    result = st.overlap_reduction_function(
        freqs, det_1, det_2, det1_name=det1_name, det2_name=det2_name
    )
    return result[f"{det1_name}-{det2_name}"]


def _upper_triangle_ets_keys(result: dict[str, np.ndarray]) -> list[str]:
    upper_keys: list[str] = []
    for key in result:
        match = _ETS_KEY_RE.match(key)
        if match is None:
            continue
        i = int(match.group(1))
        j = int(match.group(2))
        if i < j:
            upper_keys.append(key)
    upper_keys.sort()
    if not upper_keys:
        raise ValueError("No ETS upper-triangle keys found in gwfast ORF output")
    return upper_keys


def _save_hlv_fixture() -> None:
    data: dict[str, np.ndarray] = {"frequencies": FREQUENCIES}
    for det1, det2 in HLV_PAIRS:
        key = f"{det1}_{det2}"
        print(f"Computing ORF for {det1}-{det2}...")
        data[key] = gwfast_orf(det1, det2, FREQUENCIES)
    HLV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(HLV_OUT_PATH, **data)
    print(f"Saved HLV fixture to {HLV_OUT_PATH}")


def _save_et_fixture() -> None:
    ets = detectors["ETS"]
    result = st.overlap_reduction_function(
        FREQUENCIES,
        ets,
        ets,
        det1_name="ETS",
        det2_name="ETS",
        arm_length_1=ET_ARM_LENGTH_KM,
        arm_length_2=ET_ARM_LENGTH_KM,
    )
    upper_keys = _upper_triangle_ets_keys(result)
    sum_upper_pairs = np.sum([result[key] for key in upper_keys], axis=0)
    data: dict[str, np.ndarray] = {
        "frequencies": FREQUENCIES,
        "sum_upper_pairs": sum_upper_pairs,
    }
    ET_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ET_OUT_PATH, **data)
    print(f"Saved ET fixture to {ET_OUT_PATH}")
    print(f"  Included keys: {', '.join(upper_keys)}")


def main() -> None:
    _save_hlv_fixture()
    _save_et_fixture()
    print("\nFixture generation completed.")


if __name__ == "__main__":
    main()
