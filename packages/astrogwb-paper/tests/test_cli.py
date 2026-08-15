from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from astrogwb_paper.cli.workflow import build_mcmc_argv, build_paper_argv, parse_args
from astrogwb_paper.paths import paper_project_root

COMMANDS = (
    "astrogwb-run-mcmc",
    "astrogwb-validate-config",
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


# --------------------------------------------------------------------------- #
# Workflow argv construction
# --------------------------------------------------------------------------- #
def _config_groups(argv: list[str]) -> list[list[str]]:
    """Return the KEY=VALUE payload of each ``--config`` occurrence in argv."""
    groups: list[list[str]] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config":
            i += 1
            entries: list[str] = []
            while i < len(argv) and not argv[i].startswith("-"):
                entries.append(argv[i])
                i += 1
            groups.append(entries)
            continue
        i += 1
    return groups


def test_mcmc_argv_maps_experiment_to_complete_target() -> None:
    argv = build_mcmc_argv(["H0-all-detectors"], "slurm-cpu")

    groups = _config_groups(argv)
    assert len(groups) == 1
    assert "jax_platforms=cpu" in groups[0]
    assert "H0_all_detectors" in argv


def test_mcmc_argv_maps_chains_only_target() -> None:
    argv = build_mcmc_argv(
        ["modified-propagation-all-detectors"],
        "slurm",
        chains_only=True,
    )

    assert "modified_propagation_all_detectors_chains" in argv


def test_mcmc_argv_omits_config_when_nothing_needs_setting() -> None:
    assert _config_groups(build_mcmc_argv(None, "slurm")) == []


def test_mcmc_argv_merges_caller_supplied_config_entries() -> None:
    argv = build_mcmc_argv(
        ["H0-all-detectors"],
        "slurm-cpu",
        extra=["--config", "catalogs_dir=/tmp/catalogs"],
    )

    groups = _config_groups(argv)
    assert len(groups) == 1
    assert set(groups[0]) == {
        "jax_platforms=cpu",
        "catalogs_dir=/tmp/catalogs",
    }


def test_paper_argv_forwards_config_after_double_hyphen() -> None:
    args = parse_args(["paper", "--", "--config", "catalogs_dir=/tmp/catalogs"])

    argv = build_paper_argv(args.target, dry_run=True, extra=args.extra)

    assert args.target is None
    assert argv[argv.index("standalone_figures") + 1 :] == [
        "--config",
        "catalogs_dir=/tmp/catalogs",
    ]
