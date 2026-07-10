"""Tests for the SLURM submission driver in scripts/submit_mcmc.py."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "submit_mcmc", REPO_ROOT / "scripts" / "submit_mcmc.py"
)
assert _SPEC is not None
_SUBMIT = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_SUBMIT)


@pytest.fixture
def fake_sbatch(monkeypatch):
    """Stub subprocess.run and record whether logs/ exists when sbatch runs."""
    calls: list[dict] = []

    def _run(cmd, **_kwargs):
        calls.append({"cmd": cmd, "logs_exists": Path("logs").is_dir()})
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Submitted batch job 1\n", stderr=""
        )

    monkeypatch.setattr(_SUBMIT.subprocess, "run", _run)
    return calls


def test_submit_array_creates_logs_dir_before_sbatch(
    tmp_path: Path, monkeypatch, fake_sbatch
) -> None:
    # SLURM opens the --output/--error files under logs/ before the batch
    # script body runs, so the directory must exist at submission time.
    monkeypatch.chdir(tmp_path)

    _SUBMIT.submit_array(Path("manifest.txt"), config_count=2)

    assert len(fake_sbatch) == 1
    assert fake_sbatch[0]["logs_exists"]
    assert "--array=0-1" in fake_sbatch[0]["cmd"]


def test_main_manifest_path_creates_logs_dir(
    tmp_path: Path, monkeypatch, fake_sbatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "run.toml"
    config.write_text("", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{config}\n", encoding="utf-8")

    _SUBMIT.main(["--manifest", str(manifest)])

    assert len(fake_sbatch) == 1
    assert fake_sbatch[0]["logs_exists"]
