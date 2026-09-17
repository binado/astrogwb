"""Fiducials, detector networks, and plot labels for the cosmological-parameters
experiment.

Stdlib-only, deliberately: this duplicates values that also live in
``config/fiducials.json`` and ``config/networks.json``, so that a script or
notebook can name a network and get its fiducials and detectors without
reaching pydantic, JAX, or a filesystem walk. ``tests/paper/test_config_constants.py``
converts the duplication into a CI failure the moment either side drifts.

Keyed by the same run names as :data:`astrogwb.paper.plotting.DETECTOR_NETWORKS`,
so the two compose directly::

    Network(name, label, NETWORK_DETECTORS[name])
"""

from __future__ import annotations

#: The ``fiducials`` table from config/fiducials.json.
FIDUCIALS: dict[str, float] = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 770.0,
    "minimum_mass": 1.0,
    "mass_width": 1.5,
}

#: Each detector-comparison network's detector list, keyed by network name.
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

#: The network used for the joint (H0, local_merger_rate) / (xi_0, xi_n)
#: grids. Every detector network gets (H0, Omega_m) and (H0, z_peak) grids.
DEFAULT_NETWORK: str = "ET-2L-aligned-CE-Hanford"

#: LaTeX axis labels for the parameters plotted by name, matching
#: scripts/mcmc_cosmological_parameters.py's VAR_LABELS and
#: scripts/mcmc_modified_propagation.py's VAR_LABELS.
PARAMETER_LABELS: dict[str, str] = {
    "H0": r"$H_0\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    "Omega_m": r"$\Omega_m$",
    "local_merger_rate": r"$\mathcal{R}_0\,[\mathrm{Gpc^{-3}\,yr^{-1}}]$",
    "xi_0": r"$\Xi_0$",
    "xi_n": r"$n$",
    "z_peak": r"$z_\mathrm{peak}$",
}

__all__ = [
    "DEFAULT_NETWORK",
    "FIDUCIALS",
    "NETWORK_DETECTORS",
    "NETWORK_EXPERIMENT",
    "PARAMETER_LABELS",
]
