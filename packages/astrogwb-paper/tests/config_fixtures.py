"""A complete raw run config for tests, assembled the way production assembles one.

These tests used to read ``configs/mcmc.example.toml``, a standalone config kept
in parallel with the real one. It is gone. The only complete configs the project
now produces are assembled from ``inputs/mcmc.base.toml`` plus one
``experiments/<name>.toml`` run overlay -- exactly what the ``assemble_config``
workflow rule writes -- so the fixture is built the same way. The tests then
exercise the config path that actually ships instead of an example that could
drift away from it.

One difference from the old example is worth knowing when reading these tests:
the base declares a prior for *every* fiducial parameter, and
``build_run_config`` filters ``priors`` down to ``sampled_params`` (plus any
marginalized amplitude parameter). A test that needs a parameter with no prior
table must therefore delete one explicitly.
"""

from __future__ import annotations

from typing import Any

from astrogwb_paper.config.experiments import experiment, overlay_for
from astrogwb_paper.config.loading import load_mapping
from astrogwb_paper.paths import paper_project_root

PAPER_ROOT = paper_project_root()
BASE_CONFIG = PAPER_ROOT / "inputs/mcmc.base.toml"

# One sampled parameter (H0) on a three-detector network: the smallest assembly
# that still carries a prior, a full [fiducials] table, and real detectors.
EXAMPLE_EXPERIMENT = "H0-all-detectors"
EXAMPLE_RUN = "ET-2L-aligned-CE-Hanford"


def example_raw() -> dict[str, Any]:
    """Return a fresh, complete raw config mapping. Callers may mutate it."""
    return overlay_for(
        experiment(EXAMPLE_EXPERIMENT),
        EXAMPLE_RUN,
        base=load_mapping(BASE_CONFIG),
    )
