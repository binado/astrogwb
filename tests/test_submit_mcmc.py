"""Tests for the SLURM submission driver in scripts/submit_mcmc.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from astrogwb.sampling.campaign import materialize_campaign

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


def _locked_campaign(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    catalog = tmp_path / "out" / "catalog.h5"
    catalog.parent.mkdir()
    catalog.write_bytes(b"catalog bytes")
    curated = tmp_path / "curated.json"
    curated.write_text(
        json.dumps(
            {
                "fiducials": {"H0": 67.66, "local_merger_rate": 161.0},
                "priors": {"H0": {"type": "uniform", "low": 20.0, "high": 140.0}},
                "catalog": {
                    "path": "out/catalog.h5",
                    "detectors": ["E1", "E2"],
                    "f_min": 2.0,
                    "f_max": 4096.0,
                },
                "cosmology": {"z_min": 0.0, "z_max": 20.0, "n_grid": 256},
                "sampler": {"num_warmup": 2, "num_samples": 2},
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "campaign.toml"
    inventory.write_text(
        "\n".join(
            [
                "version = 1",
                'campaign_id = "test-campaign"',
                "",
                "[[runs]]",
                'id = "network-a"',
                'config = "curated.json"',
            ]
        ),
        encoding="utf-8",
    )
    lock_path = tmp_path / "campaign.lock.json"
    _, configs, manifest = materialize_campaign(
        inventory,
        lock_path=lock_path,
        frozen_dir=tmp_path / "frozen",
        root=tmp_path,
    )
    return lock_path, manifest, configs


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


def test_locked_campaign_is_validated_before_submission(
    tmp_path: Path, monkeypatch, fake_sbatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lock_path, manifest, _ = _locked_campaign(tmp_path)

    _SUBMIT.main(["--manifest", str(manifest), "--campaign-lock", str(lock_path)])

    assert len(fake_sbatch) == 1


def test_locked_campaign_rejects_changed_catalog_without_sbatch(
    tmp_path: Path, monkeypatch, fake_sbatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lock_path, manifest, _ = _locked_campaign(tmp_path)
    (tmp_path / "out" / "catalog.h5").write_bytes(b"changed catalog bytes")

    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        _SUBMIT.main(["--manifest", str(manifest), "--campaign-lock", str(lock_path)])

    assert fake_sbatch == []


def test_locked_campaign_rejects_changed_frozen_config_without_sbatch(
    tmp_path: Path, monkeypatch, fake_sbatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lock_path, manifest, configs = _locked_campaign(tmp_path)
    configs[0].write_text(
        configs[0].read_text(encoding="utf-8").replace("67.66", "70.0"),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="unlocked config"):
        _SUBMIT.main(["--manifest", str(manifest), "--campaign-lock", str(lock_path)])

    assert fake_sbatch == []


def test_locked_campaign_requires_catalog_digest(
    tmp_path: Path, monkeypatch, fake_sbatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lock_path, manifest, _ = _locked_campaign(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    del lock["runs"][0]["catalog_sha256"]
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(SystemExit, match="does not pin catalog SHA-256"):
        _SUBMIT.main(["--manifest", str(manifest), "--campaign-lock", str(lock_path)])

    assert fake_sbatch == []


def test_locked_campaign_requires_complete_manifest(
    tmp_path: Path, monkeypatch, fake_sbatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lock_path, manifest, configs = _locked_campaign(tmp_path)
    manifest.write_text(f"{configs[0]}\n{configs[0]}\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="more than once"):
        _SUBMIT.main(["--manifest", str(manifest), "--campaign-lock", str(lock_path)])

    assert fake_sbatch == []


def test_bypass_locks_allows_intentional_debug_submission(
    tmp_path: Path, monkeypatch, fake_sbatch, caplog
) -> None:
    monkeypatch.chdir(tmp_path)
    lock_path, manifest, _ = _locked_campaign(tmp_path)
    (tmp_path / "out" / "catalog.h5").write_bytes(b"changed catalog bytes")

    _SUBMIT.main(
        [
            "--manifest",
            str(manifest),
            "--campaign-lock",
            str(lock_path),
            "--bypass-locks",
        ]
    )

    assert len(fake_sbatch) == 1
    assert "bypassing campaign-lock validation" in caplog.text
