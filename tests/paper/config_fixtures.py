"""A complete raw run config for tests, assembled the way production assembles one.

These tests run the same three-layer ``config/analysis/`` merge that
``assemble_config`` writes. They therefore exercise the configuration path that
ships instead of maintaining a parallel standalone example.

One difference from the old example is worth knowing when reading these tests:
the base declares and ``build_run_config`` retains a prior for *every* fiducial
parameter. A test that needs a parameter with no prior table must therefore
delete one explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from astrogwb.paper.config.runs import ROOT_LAYERS, assemble_run

# One sampled parameter (H0) on a three-detector network: the smallest assembly
# that still carries a prior, a full [fiducials] table, and real detectors.
EXAMPLE_EXPERIMENT = "cosmological-parameters"
EXAMPLE_RUN = "ET-2L-aligned-CE-Hanford"


def example_raw() -> dict[str, Any]:
    """Return a fresh, complete raw config mapping. Callers may mutate it."""
    return assemble_run(EXAMPLE_EXPERIMENT, EXAMPLE_RUN)


def write_root_layers(
    root: Path,
    *,
    fiducials: dict[str, float] | None = None,
    priors: dict[str, Any] | None = None,
    networks: dict[str, list[str]] | None = None,
) -> None:
    """Write minimal ``config/*.json`` layer-0 files into a scratch tree.

    Every path helper goes through `base_config_paths`, which requires layer 0,
    so a tmp_path tree that exercises the run/catalog layers needs these three
    files to exist even when the test says nothing about them. Defaults are the
    smallest mappings that parse; pass a table explicitly when the test is
    about its content.
    """
    tables: dict[str, Any] = {
        "fiducials": fiducials if fiducials is not None else {"H0": 67.66},
        "priors": priors
        if priors is not None
        else {"H0": {"dist": "Uniform", "kwargs": {"low": 20.0, "high": 140.0}}},
        "networks": networks if networks is not None else {"demo": ["S1", "R1"]},
    }
    for path in ROOT_LAYERS:
        key = path.stem
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({key: tables[key]}), encoding="utf-8")
