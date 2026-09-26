"""A complete raw run config for tests, assembled the way production assembles one.

These tests run the same layered ``config/`` merge the workflow runs. They
therefore exercise the configuration path that ships instead of maintaining a
parallel standalone example.

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


#: The smallest ``[analysis]`` and ``[sampler]`` blocks that validate. Neither
#: is what any test using them is about -- they exist because
#: `base_config_paths` requires every shared layer to be present.
_MINIMAL_ANALYSIS: dict[str, Any] = {
    "minimum_frequency": 2.0,
    "maximum_frequency": 2048.0,
    "population": {
        "model_kwargs": {
            "minimum_redshift": 0.3,
            "maximum_redshift": 20.0,
            "n_grid": 256,
        }
    },
    "catalog": {
        "injection": {"seed": 1, "num_samples": 8},
        "proposal": {"seed": 2, "num_samples": 8},
    },
}
_MINIMAL_SAMPLER: dict[str, Any] = {"num_warmup": 2, "num_samples": 4}
_MINIMAL_WAVEFORM: dict[str, Any] = {
    "approximant": "AnalyticInspiral",
    "minimum_frequency": 10.0,
    "maximum_frequency": 16.0,
    "reference_frequency": 10.0,
    "sampling_frequency": 64.0,
    "frequency_resolution": 2.0,
}
_MINIMAL_POPULATION: dict[str, Any] = {
    "model_name": "bns_md_cosmological",
    "model_kwargs": {"minimum_redshift": 0.0, "maximum_redshift": 20.0, "n_grid": 64},
}


def write_root_layers(
    root: Path,
    *,
    analysis: dict[str, Any] | None = None,
    fiducials: dict[str, float] | None = None,
    networks: dict[str, list[str]] | None = None,
    priors: dict[str, Any] | None = None,
    sampler: dict[str, Any] | None = None,
    waveform: dict[str, Any] | None = None,
    population: dict[str, Any] | None = None,
) -> None:
    """Write minimal shared ``config/*.json`` layers into a scratch tree.

    Every path helper goes through `base_config_paths`, which requires all of
    :data:`ROOT_LAYERS`, so a tmp_path tree that exercises the run or catalog
    layers needs each one to exist even when the test says nothing about it.
    Defaults are the smallest mappings that parse; pass a block explicitly when
    the test is about its content.

    Each file is keyed by its own stem, which is the layer-0 convention rather
    than a convenience here: a test tree that broke it would not be exercising
    the real one.
    """
    tables: dict[str, Any] = {
        "analysis": analysis if analysis is not None else _MINIMAL_ANALYSIS,
        "fiducials": fiducials if fiducials is not None else {"H0": 67.66},
        "networks": networks if networks is not None else {"demo": ["S1", "R1"]},
        "priors": priors
        if priors is not None
        else {"H0": {"dist": "Uniform", "kwargs": {"low": 20.0, "high": 140.0}}},
        "sampler": sampler if sampler is not None else _MINIMAL_SAMPLER,
        "waveform": waveform if waveform is not None else _MINIMAL_WAVEFORM,
        "population": population if population is not None else _MINIMAL_POPULATION,
    }
    for path in ROOT_LAYERS:
        key = path.stem
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({key: tables[key]}), encoding="utf-8")
