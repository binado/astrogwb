from __future__ import annotations

import subprocess
import sys

from repo import REPO_ROOT


def test_paper_imports_leave_the_xla_backend_uninitialized() -> None:
    """Importing jax-touching paper modules must not initialize the XLA backend.

    ``import jax`` / ``import jax.numpy`` only load the package. Backend init
    (``jax.devices()``, array creation) is what freezes ``JAX_PLATFORMS`` /
    ``set_host_device_count``. A late ``set_host_device_count(2)`` still yielding
    two devices proves the imports and the config accessors did not consume that
    configuration. Loading the shared configuration must leave the backend free.
    """
    code = """
from pathlib import Path
import astrogwb.background.catalog
import astrogwb.paper.posteriors
from astrogwb.config import load_config
from astrogwb.detector.config import load_detector_config
load_detector_config([Path('config/detectors.toml')])
assert len(load_config().priors) == 10

import numpyro

numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"backend initialized early: {jax.devices()}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_detector_and_shared_config_import_in_either_order() -> None:
    """The shared config and detector registry must not form an import cycle."""
    for imports in (
        "import astrogwb.config; import astrogwb.detector.config",
        "import astrogwb.detector.config; import astrogwb.config",
    ):
        result = subprocess.run(
            [sys.executable, "-c", imports],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
