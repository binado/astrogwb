from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMIT = REPO_ROOT / "scripts" / "submit_mcmc.sh"


def _mock_sbatch(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    output = tmp_path / "sbatch-args.txt"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"${SBATCH_ARGS}\"\ncat >/dev/null\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["SBATCH_ARGS"] = str(output)
    return environment, output


def test_manifest_mode_submits_supplied_deterministic_manifest(tmp_path: Path) -> None:
    config = tmp_path / "run.json"
    config.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "array-manifest.txt"
    manifest.write_text(f"{config}\n", encoding="utf-8")
    environment, sbatch_args = _mock_sbatch(tmp_path)

    result = subprocess.run(
        [str(SUBMIT), "--manifest", str(manifest)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert f"Manifest: {manifest}" in result.stdout
    assert "--array=0-0" in sbatch_args.read_text(encoding="utf-8")


@pytest.mark.parametrize("content, error", [("", "is empty"), ("bad.txt\n", "not a TOML")])
def test_manifest_mode_rejects_empty_or_malformed_manifest(
    tmp_path: Path, content: str, error: str
) -> None:
    manifest = tmp_path / "bad-manifest.txt"
    manifest.write_text(content, encoding="utf-8")

    result = subprocess.run(
        [str(SUBMIT), "--manifest", str(manifest)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert error in result.stderr


def test_directory_mode_remains_supported(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "run.json").write_text("{}", encoding="utf-8")
    environment, sbatch_args = _mock_sbatch(tmp_path)

    subprocess.run(
        [str(SUBMIT), "--input", str(config_dir)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--array=0-0" in sbatch_args.read_text(encoding="utf-8")
