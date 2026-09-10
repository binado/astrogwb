"""Guard against drift between astrogwb.paper.config.constants and the TOML.

astrogwb.paper.config.constants duplicates values that are otherwise only
assembled by merging config layers -- that duplication is the cost of a
notebook being able to name a network and get its fiducials/detectors without
touching pydantic or JAX. These tests turn drift into a CI failure instead of
a silent mismatch.
"""

from __future__ import annotations

from repo import REPO_ROOT

from astrogwb.paper.config.constants import (
    DEFAULT_NETWORK,
    FIDUCIALS,
    NETWORK_DETECTORS,
    NETWORK_EXPERIMENT,
    PARAMETER_LABELS,
)
from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.config.runs import assemble_run
from astrogwb.paper.plotting import DETECTOR_NETWORK_RUNS, DETECTOR_NETWORKS


def test_fiducials_match_the_default_network_run_config() -> None:
    config = build_run_config(assemble_run(NETWORK_EXPERIMENT, DEFAULT_NETWORK))

    assert FIDUCIALS == config.fiducials


def test_network_detectors_match_each_run_config() -> None:
    for name, _ in DETECTOR_NETWORKS:
        merged = assemble_run(NETWORK_EXPERIMENT, name)
        expected = tuple(merged["networks"][merged["analysis"]["network"]])

        assert NETWORK_DETECTORS[name] == expected


def test_network_detectors_keys_match_detector_network_runs() -> None:
    assert tuple(NETWORK_DETECTORS) == DETECTOR_NETWORK_RUNS


def test_parameter_labels_keys_are_known_fiducials() -> None:
    """A label for a renamed/removed parameter would be exactly this failure mode."""
    assert set(PARAMETER_LABELS) <= set(FIDUCIALS)


def test_cosmological_grid_notebook_priors_name_every_fiducial() -> None:
    """``LogDensityFn`` pins non-gridded sites from ``FIDUCIALS`` keyed by ``PRIORS``."""
    source = (REPO_ROOT / "notebooks/cosmological_parameters_grid.py").read_text(
        encoding="utf-8"
    )
    missing = [name for name in FIDUCIALS if f'"{name}": dist.' not in source]
    assert not missing, missing
