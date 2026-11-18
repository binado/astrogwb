"""The tiny closed-form setup the simulator tests share.

An event pool of a few tens of BNS sources, the closed-form inspiral on a
four-bin grid and two networks, a three-detector one and a pair it contains:
every stage runs for real in well under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astrogwb.detector import DetectorConfig, DetectorNetworkConfig, load_detector
from astrogwb.waveform import WaveformMetadata

WAVEFORM = WaveformMetadata(
    approximant="AnalyticInspiral",
    minimum_frequency=20.0,
    maximum_frequency=50.0,
    reference_frequency=20.0,
    sampling_frequency=128.0,
    frequency_resolution=10.0,
)


def packaged_config(name: str) -> DetectorConfig:
    """The settings of a packaged runtime detector."""
    detector = load_detector(name)
    return DetectorConfig(
        geometry=detector.geometry,
        psd_reference=str(detector.sensitivity.psd_reference),
    )


TRIANGLE = DetectorNetworkConfig(
    name="triangle",
    detectors={name: packaged_config(name) for name in ("S1", "R1", "C1")},
)
PAIR = DetectorNetworkConfig(
    name="pair", detectors={name: TRIANGLE.detectors[name] for name in ("S1", "C1")}
)
NETWORKS = (TRIANGLE, PAIR)


def event_pool(size: int = 64, *, seed: int = 41) -> dict[str, np.ndarray]:
    """An isotropic sample with physical waveform inputs and unique sky positions."""
    rng = np.random.default_rng(seed)
    masses = np.sort(rng.uniform(1, 2.5, (2, size)), axis=0)
    return {
        "source_frame_mass_1": masses[1],
        "source_frame_mass_2": masses[0],
        "spin_1z": np.zeros(size),
        "spin_2z": np.zeros(size),
        "lambda_1": np.full(size, 400.0),
        "lambda_2": np.full(size, 400.0),
        "redshift": np.full(size, 0.1),
        "luminosity_distance": np.full(size, 450.0),
        "inclination": np.arccos(rng.uniform(-1, 1, size)),
        "declination": np.arcsin(rng.uniform(-1, 1, size)),
        "right_ascension": rng.uniform(0, 2 * np.pi, size),
        "polarization_angle": rng.uniform(0, np.pi, size),
        "coa_phase": rng.uniform(0, 2 * np.pi, size),
    }


def write_population_csv(path: Path, size: int = 64, *, seed: int = 41) -> Path:
    """The event pool as a gwmock-pop catalogue: the luminosity distance is ``distance``."""
    pool = event_pool(size, seed=seed)
    pool["distance"] = pool.pop("luminosity_distance")
    names = list(pool)
    np.savetxt(
        path,
        np.column_stack([pool[name] for name in names]),
        delimiter=",",
        header=",".join(names),
        comments="",
    )
    return path
