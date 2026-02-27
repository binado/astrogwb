# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gwfast",
#   "numpy<1.27",
# ]
# ///
"""Generate GWFast ORF reference fixtures.

Run once in a throwaway environment via::

    uv run scripts/generate_orf_fixtures.py

Saves ``tests/fixtures/gwfast_orf_reference.npz`` with arrays:
``frequencies``, ``H1_L1``, ``H1_V1`` (50 points, 20–2048 Hz geomspace grid).
"""

from __future__ import annotations

import pathlib

import numpy as np
from gwfast.signal import GWSignal

PAIRS = [("H1", "L1"), ("H1", "V1")]
FREQUENCIES = np.geomspace(20, 2048, 50)
OUT_PATH = (
    pathlib.Path(__file__).parent.parent
    / "tests"
    / "fixtures"
    / "gwfast_orf_reference.npz"
)


def gwfast_orf(det1_name: str, det2_name: str, freqs: np.ndarray) -> np.ndarray:
    signal = GWSignal(
        wf_model=None,
        psd_path=None,
        detector_name=det1_name,
        network=False,
    )
    return signal.ORF(det2_name, freqs)


def main() -> None:
    data: dict[str, np.ndarray] = {"frequencies": FREQUENCIES}
    for det1, det2 in PAIRS:
        key = f"{det1}_{det2}"
        print(f"Computing ORF for {det1}-{det2}...")
        data[key] = gwfast_orf(det1, det2, FREQUENCIES)
        print(f"  {key}: {data[key][:3]}...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **data)
    print(f"\nSaved fixture to {OUT_PATH}")


if __name__ == "__main__":
    main()
