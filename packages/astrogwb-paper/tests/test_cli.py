from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from astrogwb_paper.paths import paper_project_root

COMMANDS = (
    "astrogwb-run-mcmc",
    "astrogwb-generate-mcmc-configs",
    "astrogwb-generate-waveform-catalog",
    "astrogwb-profile-model",
    "astrogwb-workflow",
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


def test_config_and_runtime_imports_do_not_initialize_jax() -> None:
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
