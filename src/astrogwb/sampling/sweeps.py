"""Declarative campaign matrix shared by config generation and Snakemake."""

from __future__ import annotations

from typing import Any

FIDUCIAL_H0 = 67.66
FIDUCIAL_LOCAL_MERGER_RATE = 161.0
RELATIVE_GAUSSIAN_SIGMA = 0.01

DETECTOR_NETWORKS: dict[str, tuple[str, ...]] = {
    "ET-triangular": ("E1", "E2", "E3"),
    "ET-triangular-CE-Hanford": ("E1", "E2", "E3", "C1"),
    "ET-2L-aligned": ("S1", "R1"),
    "ET-2L-aligned-CE-Hanford": ("S1", "R1", "C1"),
    "ET-2L-misaligned": ("S2", "R2"),
    "ET-2L-misaligned-CE-Hanford": ("S2", "R2", "C1"),
}

PRIOR_TABLES: dict[str, dict[str, Any]] = {
    "H0": {"type": "uniform", "low": 20.0, "high": 140.0},
    "Omega_m": {"type": "uniform", "low": 0.05, "high": 0.95},
    "xi_0": {"type": "uniform", "low": 0.5, "high": 5.0},
    "xi_n": {"type": "uniform", "low": 0.3, "high": 3.0},
    "gamma": {"type": "uniform", "low": 0.5, "high": 10.0},
    "kappa": {"type": "uniform", "low": 0.05, "high": 10.0},
    "z_peak": {"type": "uniform", "low": 0.05, "high": 10.0},
    "local_merger_rate": {"type": "uniform", "low": 7.6, "high": 250.0},
}

CAMPAIGNS: dict[str, dict[str, tuple[tuple[str, ...], dict[str, dict[str, Any]]]]] = {
    "cosmology": {
        "H0": (("H0",), {}),
        "H0-Omega_m": (("H0", "Omega_m"), {}),
        "H0-merger-rate": (("H0", "local_merger_rate"), {}),
        "H0-merger-rate-gauss": (
            ("H0", "local_merger_rate"),
            {
                "local_merger_rate": {
                    "type": "normal",
                    "loc": FIDUCIAL_LOCAL_MERGER_RATE,
                    "scale": RELATIVE_GAUSSIAN_SIGMA * FIDUCIAL_LOCAL_MERGER_RATE,
                },
            },
        ),
    },
    "modified-propagation": {
        "Xi_0": (("xi_0",), {}),
        "Xi_0-n": (("xi_0", "xi_n"), {}),
        "Xi_0-H0-gauss": (
            ("xi_0", "H0"),
            {
                "H0": {
                    "type": "normal",
                    "loc": FIDUCIAL_H0,
                    "scale": RELATIVE_GAUSSIAN_SIGMA * FIDUCIAL_H0,
                },
            },
        ),
    },
    "astrophysical": {
        "H0-peak": (("H0", "z_peak"), {}),
        "H0-MD": (("H0", "gamma", "kappa", "z_peak"), {}),
        "Xi_0-MD": (("xi_0", "gamma", "kappa", "z_peak"), {}),
    },
}


def sweep_filenames() -> list[str]:
    """Return ``{campaign}/{network}__{sample}.json`` paths."""
    return [
        f"{campaign}/{network_label}__{sample_label}.json"
        for campaign, sample_sets in CAMPAIGNS.items()
        for network_label in DETECTOR_NETWORKS
        for sample_label in sample_sets
    ]


def campaign_runs(campaign: str) -> list[str]:
    """Return sorted run stems for one generated campaign."""
    return sorted(
        filename.removeprefix(f"{campaign}/").removesuffix(".json")
        for filename in sweep_filenames()
        if filename.startswith(f"{campaign}/")
    )
