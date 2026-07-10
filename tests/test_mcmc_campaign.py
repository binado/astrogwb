from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrogwb.sampling.campaign import (
    build_lock,
    config_sha256,
    file_sha256,
    load_lock,
    materialize_campaign,
)
from astrogwb.sampling.config import RunConfig


def _config(*, prior_high: float = 140.0) -> dict[str, object]:
    return {
        "fiducials": {"H0": 67.66, "local_merger_rate": 161.0},
        "priors": {"H0": {"type": "uniform", "low": 20.0, "high": prior_high}},
        "catalog": {
            "path": "out/catalog.h5",
            "detectors": ["E1", "E2"],
            "f_min": 2.0,
            "f_max": 4096.0,
        },
        "cosmology": {"z_min": 0.0, "z_max": 20.0, "n_grid": 256},
        "sampler": {"num_warmup": 2, "num_samples": 2},
    }


def _inventory(tmp_path: Path, *, legacy: bool = False) -> Path:
    catalog = tmp_path / "out" / "catalog.h5"
    catalog.parent.mkdir()
    catalog.write_bytes(b"catalog bytes")
    curated = tmp_path / "curated.json"
    curated.write_text(json.dumps(_config()), encoding="utf-8")
    inventory = tmp_path / "campaign.toml"
    inventory.write_text(
        "\n".join(
            [
                "version = 1",
                'campaign_id = "test-campaign"',
                f"legacy = {str(legacy).lower()}",
                "",
                "[[runs]]",
                'id = "network-a"',
                'config = "curated.json"',
            ]
        ),
        encoding="utf-8",
    )
    return inventory


def test_lock_uses_canonical_resolved_config_and_deterministic_outputs(
    tmp_path: Path,
) -> None:
    inventory = _inventory(tmp_path)

    lock = build_lock(inventory, root=tmp_path)

    run = lock.run("network-a")
    assert run.outputs == {
        "chain": "chains/test-campaign/network-a.nc",
        "sidecar": "chains/test-campaign/network-a.json",
    }
    config = RunConfig.model_validate(run.config)
    assert config.priors["H0"]["high"] == 140.0
    assert config.sampler.dense_mass is True
    assert run.config_sha256 == config_sha256(config)
    assert run.catalog_sha256 == file_sha256(tmp_path / "out" / "catalog.h5")
    assert build_lock(inventory, root=tmp_path) == lock


def test_inventory_rejects_duplicate_ids_and_paths(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    inventory.write_text(
        inventory.read_text(encoding="utf-8")
        + '\n[[runs]]\nid = "network-a"\nconfig = "curated.json"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate run IDs"):
        build_lock(inventory, root=tmp_path)

    inventory.write_text(
        inventory.read_text(encoding="utf-8").replace(
            'id = "network-a"', 'id = "network-b"', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate curated config paths"):
        build_lock(inventory, root=tmp_path)


def test_freeze_rematerializes_lock_and_rejects_config_changes(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    lock_path = tmp_path / "campaign.lock.json"
    frozen_dir = tmp_path / "frozen"
    lock, configs, manifest = materialize_campaign(
        inventory, lock_path=lock_path, frozen_dir=frozen_dir, root=tmp_path
    )
    assert configs == [frozen_dir / "network-a.json"]
    assert manifest.read_text(encoding="utf-8") == "frozen/network-a.json\n"

    configs[0].unlink()
    rematerialized, configs, _ = materialize_campaign(
        inventory, lock_path=lock_path, frozen_dir=frozen_dir, root=tmp_path
    )
    assert rematerialized == lock
    assert configs[0].exists()
    assert load_lock(lock_path) == lock

    curated = tmp_path / "curated.json"
    curated.write_text(json.dumps(_config(prior_high=120.0)), encoding="utf-8")
    with pytest.raises(ValueError, match="create a new campaign ID"):
        materialize_campaign(inventory, lock_path=lock_path, root=tmp_path)


def test_legacy_inventory_preserves_explicit_outputs(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path, legacy=True)
    curated = tmp_path / "curated.json"
    data = _config()
    data["output"] = {"outdir": "chains", "label": "old-timestamped-name"}
    curated.write_text(json.dumps(data), encoding="utf-8")

    lock = build_lock(inventory, root=tmp_path)
    run = lock.run("network-a")
    assert run.outputs["chain"] == "chains/old-timestamped-name.nc"
    assert run.config["fiducials"]["local_merger_rate"] == 161.0
    assert run.config["sampler"]["max_tree_depth"] == 10


def test_freeze_rejects_catalog_changes(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    lock_path = tmp_path / "campaign.lock.json"
    materialize_campaign(inventory, lock_path=lock_path, root=tmp_path)

    (tmp_path / "out" / "catalog.h5").write_bytes(b"changed catalog bytes")

    with pytest.raises(ValueError, match="catalog for run 'network-a' changed"):
        materialize_campaign(inventory, lock_path=lock_path, root=tmp_path)


def test_historical_lock_without_catalog_digest_remains_readable() -> None:
    lock = load_lock(
        Path(__file__).resolve().parent.parent
        / "configs/mcmc/campaigns/paper-h0-legacy.lock.json"
    )

    assert all(run.catalog_sha256 is None for run in lock.runs)
