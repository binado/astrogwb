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

from pathlib import Path
from typing import Any

import tomli_w

from astrogwb.paper.config.runs import (
    DEFAULTS_PATH,
    DETECTORS_PATH,
    POPULATIONS_PATH,
    WAVEFORMS_PATH,
    assemble_run,
)

# One sampled parameter (H0) on a three-detector network: the smallest assembly
# that still carries a prior, a full [fiducials] table, and real detectors.
EXAMPLE_EXPERIMENT = "cosmological-parameters"
EXAMPLE_RUN = "ET-2L-aligned-CE-Hanford"


def example_raw() -> dict[str, Any]:
    """Return a fresh, complete raw config mapping. Callers may mutate it."""
    return assemble_run(EXAMPLE_EXPERIMENT, EXAMPLE_RUN)


def _role() -> dict[str, Any]:
    """A catalog role over the default waveform and population."""
    return {
        "waveform": "${waveforms.default}",
        "fiducials": "${fiducials}",
        "num_samples": 8,
        "population": "${populations.default}",
    }


#: The smallest ``[analysis]`` and ``[sampler]`` blocks that validate. Neither
#: is what any test using them is about -- they exist because
#: `base_config_paths` requires a complete shared layer.
_MINIMAL_ANALYSIS: dict[str, Any] = {
    "minimum_frequency": 2.0,
    "maximum_frequency": 2048.0,
    "population": "${populations.target}",
    "injection": _role(),
    "proposal": _role(),
    "seeds": {"injection": 1, "proposal": 2},
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


def _minimal_population(minimum_redshift: float, n_grid: int) -> dict[str, Any]:
    return {
        "model_name": "bns_madau_dickinson",
        "model_kwargs": {},
        "redshift": {
            "model": "madau_dickinson",
            "kwargs": {
                "minimum_redshift": minimum_redshift,
                "maximum_redshift": 20.0,
                "n_grid": n_grid,
            },
        },
        "mass": {"model": "ordered_uniform"},
    }


_MINIMAL_POPULATIONS: dict[str, Any] = {
    "default": _minimal_population(0.0, 64),
    "target": _minimal_population(0.3, 256),
}


def write_defaults(
    root: Path,
    *,
    analysis: dict[str, Any] | None = None,
    fiducials: dict[str, float] | None = None,
    networks: dict[str, list[str]] | None = None,
    priors: dict[str, Any] | None = None,
    sampler: dict[str, Any] | None = None,
    waveform: dict[str, Any] | None = None,
    populations: dict[str, Any] | None = None,
) -> None:
    """Write the four minimal shared layers into a scratch tree.

    Every path helper goes through `base_config_paths`, which requires each
    shared layer, so a tmp_path tree that exercises the run or catalog layers
    needs every block to exist even when the test says nothing about it.
    Defaults are the smallest mappings that validate; pass a block explicitly
    when the test is about its content. ``waveform`` is written as
    ``[waveforms.default]``, and ``populations`` must name a ``default`` and a
    ``target``.
    """
    defaults = {
        "analysis": analysis if analysis is not None else _MINIMAL_ANALYSIS,
        "fiducials": fiducials if fiducials is not None else {"H0": 67.66},
        "priors": priors
        if priors is not None
        else {"H0": {"dist": "Uniform", "kwargs": {"low": 20.0, "high": 140.0}}},
        "sampler": sampler if sampler is not None else _MINIMAL_SAMPLER,
    }
    layers = {
        DEFAULTS_PATH: defaults,
        WAVEFORMS_PATH: {
            "waveforms": {
                "default": waveform if waveform is not None else _MINIMAL_WAVEFORM
            }
        },
        POPULATIONS_PATH: {
            "populations": populations
            if populations is not None
            else _MINIMAL_POPULATIONS
        },
        DETECTORS_PATH: {
            "networks": networks if networks is not None else {"demo": ["S1", "R1"]},
            "detectors": {},
        },
    }
    for path, tables in layers.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(tomli_w.dumps(tables), encoding="utf-8")
