from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from astrogwb_paper.paths import paper_project_root

COMMANDS = (
    "astrogwb-generate-population",
    "astrogwb-run-mcmc",
    "astrogwb-validate-config",
    "astrogwb-generate-waveform-catalog",
    "astrogwb-profile-model",
)


@pytest.mark.parametrize("command", COMMANDS)
def test_console_command_help_from_nested_directory(
    command: str, tmp_path: Path
) -> None:
    result = subprocess.run(
        [command, "--help"],
        cwd=paper_project_root() / "notebooks",
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_help_path_imports_do_not_import_jax() -> None:
    """CLI --help modules must not pull jax into sys.modules.

    ``astrogwb-run-mcmc --help`` imports config and runtime at module load.
    Importing jax is slow; it is not needed to parse flags. Catalogs, inference,
    and snr are allowed to import jax -- they are not on this graph.
    """
    code = """
import sys
import astrogwb_paper
import astrogwb_paper.config.analysis
import astrogwb_paper.config.mcmc
import astrogwb_paper.config.figures
import astrogwb_paper.runtime
assert 'jax' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=paper_project_root() / "notebooks",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_catalog_inference_snr_imports_leave_the_xla_backend_uninitialized() -> None:
    """Importing jax-touching paper modules must not initialize the XLA backend.

    ``import jax`` / ``import jax.numpy`` only load the package. Backend init
    (``jax.devices()``, array creation) is what freezes ``JAX_PLATFORMS`` /
    ``set_host_device_count``. A late ``set_host_device_count(2)`` still yielding
    two devices proves catalogs / inference / snr did not consume that config.
    """
    code = """
import astrogwb_paper.catalogs
import astrogwb_paper.inference
import astrogwb_paper.snr
import numpyro

numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"backend initialized early: {jax.devices()}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=paper_project_root() / "notebooks",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
