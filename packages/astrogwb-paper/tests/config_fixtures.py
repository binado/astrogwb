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

from typing import Any

from astrogwb_paper.config.runs import assemble_run

# One sampled parameter (H0) on a three-detector network: the smallest assembly
# that still carries a prior, a full [fiducials] table, and real detectors.
EXAMPLE_EXPERIMENT = "cosmological-parameters"
EXAMPLE_RUN = "ET-2L-aligned-CE-Hanford"


def example_raw() -> dict[str, Any]:
    """Return a fresh, complete raw config mapping. Callers may mutate it."""
    return assemble_run(EXAMPLE_EXPERIMENT, EXAMPLE_RUN)
