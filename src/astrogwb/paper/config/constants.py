"""Fiducials and detector networks for the cosmological-parameters experiment.

Stdlib-only, deliberately: this duplicates values that also live in
``config/analysis/base/parameters.toml`` and the per-network run TOMLs under
``config/analysis/runs/cosmological-parameters/``, so that a script or
notebook can name a network and get its fiducials and detectors without
reaching pydantic, JAX, or a filesystem walk. ``tests/paper/test_config_constants.py``
converts the duplication into a CI failure the moment either side drifts.

Keyed by the same run names as :data:`astrogwb.paper.plotting.DETECTOR_NETWORKS`,
so the two compose directly::

    Network(name, label, NETWORK_DETECTORS[name])
"""

from __future__ import annotations

#: The [fiducials] table from config/analysis/base/parameters.toml.
FIDUCIALS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
}

#: Each detector-comparison network's [analysis.detectors], keyed by run name.
#: Not "DETECTOR_NETWORKS" -- that name is already taken by
#: astrogwb.paper.plotting.DETECTOR_NETWORKS, which holds (run name, label)
#: pairs and carries no detectors.
NETWORK_DETECTORS: dict[str, tuple[str, ...]] = {
    "ET-triangular": ("E1", "E2", "E3"),
    "ET-triangular-CE-Hanford": ("E1", "E2", "E3", "C1"),
    "ET-2L-aligned": ("S1", "R1"),
    "ET-2L-aligned-CE-Hanford": ("S1", "R1", "C1"),
    "ET-2L-misaligned": ("S2", "R2"),
    "ET-2L-misaligned-CE-Hanford": ("S2", "R2", "C1"),
}

#: The experiment every constant here was read off of.
NETWORK_EXPERIMENT: str = "cosmological-parameters"

#: The network used for the joint (H0, Omega_m) / (H0, local_merger_rate) grids.
DEFAULT_NETWORK: str = "ET-2L-aligned-CE-Hanford"

__all__ = [
    "DEFAULT_NETWORK",
    "FIDUCIALS",
    "NETWORK_DETECTORS",
    "NETWORK_EXPERIMENT",
]
