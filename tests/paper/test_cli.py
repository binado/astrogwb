from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from astrogwb.paper.paths import paper_project_root

COMMANDS = (
    "astrogwb-generate-bank",
    "astrogwb-run-mcmc",
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

    ``astrogwb.paper.config.mcmc`` and ``astrogwb.paper.runtime`` parse flags
    and configure the process without touching jax: importing jax is slow, and
    ``runtime`` deliberately imports it only inside ``configure_runtime``.

    ``config.runs`` and ``config.banks`` are in this list rather than excluded
    from it, and that is the point: the ``Snakefile`` imports both to build the
    DAG, so every ``--dry-run`` paid for a jax import while ``config.banks``
    read ``PopulationMetadata`` from ``astrogwb.catalog`` at module scope. It
    reads it inside the two functions that need it now, and ``config.runs``
    reaches no further than stdlib.
    """
    code = """
import sys
import astrogwb.paper
import astrogwb.paper.config.mcmc
import astrogwb.paper.config.runs
import astrogwb.paper.config.banks
import astrogwb.paper.runtime
assert 'jax' not in sys.modules
assert 'pydantic' not in sys.modules or 'astrogwb.paper.config.mcmc' in sys.modules
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
import astrogwb.paper.catalogs
import astrogwb.paper.config.banks
import astrogwb.paper.inference
import astrogwb.paper.snr
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
