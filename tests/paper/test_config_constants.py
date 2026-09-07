"""Guard against drift between astrogwb.paper.config.constants and the TOML.

astrogwb.paper.config.constants duplicates values that are otherwise only
assembled by merging config layers -- that duplication is the cost of a
notebook being able to name a network and get its fiducials/detectors without
touching pydantic or JAX. notebooks/mcmc_plotting.py:59 already carried a
stale FIDUCIALS using the pre-rename chi0/chin keys; these tests turn that
failure mode into a CI failure instead of a silent drift.
"""

from __future__ import annotations

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
        expected = tuple(
            assemble_run(NETWORK_EXPERIMENT, name)["analysis"]["detectors"]
        )

        assert NETWORK_DETECTORS[name] == expected


def test_network_detectors_keys_match_detector_network_runs() -> None:
    assert tuple(NETWORK_DETECTORS) == DETECTOR_NETWORK_RUNS


def test_parameter_labels_keys_are_known_fiducials() -> None:
    """A label for a renamed/removed parameter would be exactly this failure mode."""
    assert set(PARAMETER_LABELS) <= set(FIDUCIALS)
