from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from astrogwb_paper.paths import paper_project_root

COMMANDS = (
    "astrogwb-generate-bank",
    "astrogwb-run-mcmc",
    "astrogwb-assemble-config",
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


def test_pure_config_imports_do_not_import_jax() -> None:
    """Pure-config and runtime modules must not pull jax into sys.modules.

    ``astrogwb_paper.config.mcmc`` and ``astrogwb_paper.runtime`` parse flags
    and configure the process without touching jax: importing jax is slow, and
    ``runtime`` deliberately imports it only inside ``configure_runtime``.

    ``config.banks`` (and ``config.runs`` / ``config.figures`` through it) now
    imports jax transitively: it reads ``PopulationMetadata`` from
    ``astrogwb.catalog``, which re-exports waveform-grid metadata from
    ``astrogwb.waveform``. That import is accepted -- ``import jax`` only loads
    the package and does not consume runtime configuration; the sibling test
    below proves the XLA backend stays uninitialized on that path.
    """
    code = """
import sys
import astrogwb_paper
import astrogwb_paper.config.mcmc
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
    ``config.banks`` belongs on this path too: run_mcmc resolves the proposal
    density from bank attributes *before* ``configure_runtime``.
    """
    code = """
import astrogwb_paper.catalogs
import astrogwb_paper.config.banks
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
