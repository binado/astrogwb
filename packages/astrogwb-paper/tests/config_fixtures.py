"""A complete raw run config for tests, assembled the way production assembles one.

These tests build the same base-plus-run merge that ``assemble_config`` writes
from ``inputs/experiments.yaml``. They therefore exercise the configuration path
that ships instead of maintaining a parallel standalone example.

One difference from the old example is worth knowing when reading these tests:
the base declares a prior for *every* fiducial parameter, and
``build_run_config`` filters ``priors`` down to ``sampled_params`` (plus any
marginalized amplitude parameter). A test that needs a parameter with no prior
table must therefore delete one explicitly.
"""

from __future__ import annotations

from typing import Any

from astrogwb_paper.config.experiments import experiment, load_base, overlay_for

# One sampled parameter (H0) on a three-detector network: the smallest assembly
# that still carries a prior, a full [fiducials] table, and real detectors.
EXAMPLE_EXPERIMENT = "cosmological-parameters"
EXAMPLE_RUN = "ET-2L-aligned-CE-Hanford"


def example_raw() -> dict[str, Any]:
    """Return a fresh, complete raw config mapping. Callers may mutate it."""
    return overlay_for(
        experiment(EXAMPLE_EXPERIMENT),
        EXAMPLE_RUN,
        base=load_base(),
    )
