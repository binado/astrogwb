from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from astrogwb.hashing import file_sha256
from astrogwb.sampling.config import build_run_config, config_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "run_mcmc", REPO_ROOT / "scripts" / "run_mcmc.py"
)
assert _SPEC is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_RUNNER)


def _config(tmp_path: Path, **catalog_overrides):
    return build_run_config(
        {
            "fiducials": {"H0": 67.66},
            "priors": {"H0": {"type": "uniform", "low": 20.0, "high": 140.0}},
            "catalog": {
                "path": "out/catalog.h5",
                "detectors": ["E1", "E2"],
                "f_min": 2.0,
                "f_max": 4096.0,
                **catalog_overrides,
            },
            "cosmology": {"z_min": 0.0, "z_max": 20.0, "n_grid": 256},
            "sampler": {"num_warmup": 2, "num_samples": 2},
            "output": {"outdir": str(tmp_path), "label": "locked-run"},
        }
    )


def test_runner_refuses_existing_outputs_unless_forced(tmp_path: Path) -> None:
    config = _config(tmp_path)
    chain, sidecar = _RUNNER.output_paths(config)
    chain.touch()

    with pytest.raises(FileExistsError, match="--force"):
        _RUNNER.ensure_output_paths_available(config)

    assert _RUNNER.ensure_output_paths_available(config, force=True) == (chain, sidecar)


def test_build_run_record_includes_both_digests(tmp_path: Path) -> None:
    config = _config(tmp_path)

    record = _RUNNER.build_run_record(
        config, timestamp="20260710-000000", catalog_sha256="a" * 64
    )

    assert record["catalog_sha256"] == "a" * 64
    assert record["config_sha256"] == config_sha256(config)
    assert record["timestamp"] == "20260710-000000"


def _catalog_on_disk(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "catalog.h5"
    path.write_bytes(b"not really hdf5")
    return path, file_sha256(path)


def test_verify_catalog_returns_digest_without_pin(tmp_path: Path) -> None:
    path, digest = _catalog_on_disk(tmp_path)
    config = _config(tmp_path, path=str(path))

    assert _RUNNER.verify_catalog(config) == digest


def test_verify_catalog_accepts_matching_pin(tmp_path: Path) -> None:
    path, digest = _catalog_on_disk(tmp_path)
    config = _config(tmp_path, path=str(path), sha256=digest)

    assert _RUNNER.verify_catalog(config) == digest


def test_verify_catalog_rejects_stale_pin(tmp_path: Path) -> None:
    path, _ = _catalog_on_disk(tmp_path)
    config = _config(tmp_path, path=str(path), sha256="b" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _RUNNER.verify_catalog(config)


def test_verify_catalog_requires_existing_file(tmp_path: Path) -> None:
    config = _config(tmp_path, path=str(tmp_path / "missing.h5"))

    with pytest.raises(FileNotFoundError, match="catalog not found"):
        _RUNNER.verify_catalog(config)
