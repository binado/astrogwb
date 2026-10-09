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
    configuration. ``priors()`` imports numpyro but must leave the backend free.
    """
    code = """
from pathlib import Path
import astrogwb.paper.catalogs
from astrogwb.paper.config import priors
from astrogwb.paper.config.detectors import load_detector_config
load_detector_config([Path('config/detectors.toml')])
assert len(priors()) == 10

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


def test_the_accessors_hand_back_copies() -> None:
    """`@cache` returns the same object every call; the accessors must not.

    A notebook that does `p = priors(); p['H0'] = ...` would otherwise poison
    every later reader in the process. Keyword overrides are merged after the
    cached parse, so they must not reach the cache either.
    """
    from astrogwb.paper.config import fiducials, networks, priors

    fiducials(REPO_ROOT)["H0"] = 999.0
    networks(REPO_ROOT)["ET-2L-aligned"] = ("nope",)
    del priors(REPO_ROOT)["H0"]

    fiducials(REPO_ROOT, H0=1.0)
    networks(REPO_ROOT, **{"ET-2L-aligned": ("overridden",)})

    assert fiducials(REPO_ROOT)["H0"] == 67.66
    assert networks(REPO_ROOT)["ET-2L-aligned"] == ("S1", "R1")
    assert "H0" in priors(REPO_ROOT)
