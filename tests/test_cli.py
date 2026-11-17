"""CLI runtime initialization."""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_cli_leaves_the_xla_backend_uninitialized() -> None:
    """A late ``set_host_device_count(2)`` still taking effect proves it."""
    command = "model-catalog"
    code = f"""
from typer.testing import CliRunner
from astrogwb.cli import app

assert CliRunner().invoke(app, ["{command}", "--help"]).exit_code == 0
import numpyro

numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"backend initialized early: {{jax.devices()}}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
