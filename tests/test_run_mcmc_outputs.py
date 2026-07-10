from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from astrogwb.sampling.config import build_run_config

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "run_mcmc", REPO_ROOT / "scripts" / "run_mcmc.py"
)
assert _SPEC is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_RUNNER)


def _config(tmp_path: Path):
    return build_run_config(
        {
            "fiducials": {"H0": 67.66},
            "priors": {"H0": {"type": "uniform", "low": 20.0, "high": 140.0}},
            "catalog": {
                "path": "out/catalog.h5",
                "detectors": ["E1", "E2"],
                "f_min": 2.0,
                "f_max": 4096.0,
            },
            "cosmology": {"z_min": 0.0, "z_max": 20.0, "n_grid": 256},
            "sampler": {"num_warmup": 2, "num_samples": 2},
            "output": {"outdir": str(tmp_path), "label": "locked-run"},
        }
    )


def test_runner_refuses_existing_outputs_unless_forced(tmp_path: Path) -> None:
    config = _config(tmp_path)
    chain, sidecar = _RUNNER.output_paths(config)
    chain.touch()

    with pytest.raises(FileExistsError, match="--force"):
        _RUNNER.ensure_output_paths_available(config)

    assert _RUNNER.ensure_output_paths_available(config, force=True) == (chain, sidecar)
