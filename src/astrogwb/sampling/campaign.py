"""Immutable, reproducible inventories of MCMC runs.

Campaign inventories are small, editable TOML files.  Freezing an inventory turns
the validated configs into a JSON lock file.  The lock, rather than a generated
directory of configs, is the reproducibility artifact: it embeds everything the
runner needs and records the digest of every curated source config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrogwb.config import load_mapping

from .config import OutputConfig, RunConfig, build_run_config, save_config

CAMPAIGN_FORMAT_VERSION = 1
_STRICT = ConfigDict(frozen=True, extra="forbid")


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used for campaign digests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    """SHA-256 of a JSON-compatible value, independent of whitespace/key order."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def config_payload(config: RunConfig) -> dict[str, Any]:
    """Serialize a resolved config into the exact JSON form stored in a lock."""
    return config.model_dump(mode="json")


def config_sha256(config: RunConfig) -> str:
    return canonical_sha256(config_payload(config))


class CampaignRun(BaseModel):
    """One named, hand-curated MCMC configuration in an inventory."""

    model_config = _STRICT

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    config: Path


class CampaignInventory(BaseModel):
    """Editable campaign declaration."""

    model_config = _STRICT

    version: int = CAMPAIGN_FORMAT_VERSION
    campaign_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    legacy: bool = False
    runs: tuple[CampaignRun, ...]

    @model_validator(mode="after")
    def _unique_runs(self) -> CampaignInventory:
        if self.version != CAMPAIGN_FORMAT_VERSION:
            raise ValueError(
                f"unsupported campaign inventory version {self.version}; "
                f"expected {CAMPAIGN_FORMAT_VERSION}"
            )
        if not self.runs:
            raise ValueError("campaign inventory must define at least one run")
        ids = [run.id for run in self.runs]
        if len(ids) != len(set(ids)):
            raise ValueError("campaign inventory has duplicate run IDs")
        paths = [str(run.config) for run in self.runs]
        if len(paths) != len(set(paths)):
            raise ValueError("campaign inventory has duplicate curated config paths")
        return self


class LockedRun(BaseModel):
    """Frozen resolved config plus the source identity it was derived from."""

    model_config = _STRICT

    id: str
    source_config: str
    source_sha256: str
    config_sha256: str
    config: dict[str, Any]
    detectors: tuple[str, ...]
    sampled_params: tuple[str, ...]
    outputs: dict[str, str]


class CampaignLock(BaseModel):
    """The committed, self-contained immutable campaign artifact."""

    model_config = _STRICT

    version: int = CAMPAIGN_FORMAT_VERSION
    campaign_id: str
    legacy: bool = False
    runs: tuple[LockedRun, ...]

    @model_validator(mode="after")
    def _unique_runs(self) -> CampaignLock:
        if self.version != CAMPAIGN_FORMAT_VERSION:
            raise ValueError(f"unsupported campaign lock version {self.version}")
        ids = [run.id for run in self.runs]
        if len(ids) != len(set(ids)):
            raise ValueError("campaign lock has duplicate run IDs")
        return self

    def run(self, run_id: str) -> LockedRun:
        for run in self.runs:
            if run.id == run_id:
                return run
        raise KeyError(f"campaign {self.campaign_id!r} has no run {run_id!r}")


def load_inventory(path: str | Path) -> CampaignInventory:
    return CampaignInventory.model_validate(load_mapping(Path(path)))


def load_lock(path: str | Path) -> CampaignLock:
    with Path(path).open(encoding="utf-8") as handle:
        return CampaignLock.model_validate(json.load(handle))


def default_lock_path(inventory_path: str | Path) -> Path:
    path = Path(inventory_path)
    return path.with_suffix(".lock.json")


def _source_path(inventory_path: Path, config_path: Path) -> Path:
    return config_path if config_path.is_absolute() else (inventory_path.parent / config_path)


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _locked_config(
    config: RunConfig, inventory: CampaignInventory, run_id: str
) -> RunConfig:
    """Apply output convention after hashing the curated source config."""
    if inventory.legacy:
        return config
    return config.model_copy(
        update={
            "output": OutputConfig(
                outdir=Path("chains") / inventory.campaign_id,
                label=run_id,
            )
        }
    )


def build_lock(inventory_path: str | Path, *, root: str | Path | None = None) -> CampaignLock:
    """Validate an inventory and build the lock that would be committed for it."""
    inventory_file = Path(inventory_path).resolve()
    repository = Path(root).resolve() if root is not None else Path.cwd().resolve()
    inventory = load_inventory(inventory_file)
    locked_runs: list[LockedRun] = []
    for declared in inventory.runs:
        source_path = _source_path(inventory_file, declared.config).resolve()
        source = build_run_config(load_mapping(source_path))
        frozen = _locked_config(source, inventory, declared.id)
        outdir = frozen.outdir
        label = frozen.label
        if not label:
            raise ValueError(
                f"campaign run {declared.id!r} has no output label; campaign locks "
                "require stable output names"
            )
        locked_runs.append(
            LockedRun(
                id=declared.id,
                source_config=_display_path(source_path, repository),
                source_sha256=config_sha256(source),
                config_sha256=config_sha256(frozen),
                config=config_payload(frozen),
                detectors=frozen.catalog.detectors,
                sampled_params=frozen.sampled_params,
                outputs={
                    "chain": str(outdir / f"{label}.nc"),
                    "sidecar": str(outdir / f"{label}.json"),
                },
            )
        )
    return CampaignLock(
        campaign_id=inventory.campaign_id,
        legacy=inventory.legacy,
        runs=tuple(locked_runs),
    )


def write_lock(lock: CampaignLock, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _verify_source_hashes(
    lock: CampaignLock, inventory_path: Path, *, root: Path
) -> None:
    inventory = load_inventory(inventory_path)
    if inventory.campaign_id != lock.campaign_id or inventory.legacy != lock.legacy:
        raise ValueError(
            "campaign inventory no longer matches its lock; create a new campaign ID"
        )
    by_id = {run.id: run for run in inventory.runs}
    if set(by_id) != {run.id for run in lock.runs}:
        raise ValueError(
            "campaign inventory run IDs differ from the existing lock; create a new campaign ID"
        )
    for locked in lock.runs:
        declared = by_id[locked.id]
        source_path = _source_path(inventory_path, declared.config)
        if _display_path(source_path, root) != locked.source_config:
            raise ValueError(
                f"curated config path for run {locked.id!r} changed under immutable "
                f"campaign {lock.campaign_id!r}; create a new campaign ID"
            )
        source = build_run_config(load_mapping(source_path))
        if config_sha256(source) != locked.source_sha256:
            raise ValueError(
                f"curated config for run {locked.id!r} changed under immutable campaign "
                f"{lock.campaign_id!r}; create a new campaign ID"
            )


def default_frozen_dir(lock: CampaignLock, *, root: str | Path = ".") -> Path:
    return Path(root) / "configs" / "mcmc" / "frozen" / lock.campaign_id


def materialize_campaign(
    inventory_path: str | Path,
    *,
    lock_path: str | Path | None = None,
    frozen_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    root: str | Path | None = None,
) -> tuple[CampaignLock, list[Path], Path]:
    """Create a first lock or rematerialize generated configs from an existing one."""
    inventory_file = Path(inventory_path).resolve()
    repository = Path(root).resolve() if root is not None else Path.cwd().resolve()
    requested_lock = Path(lock_path) if lock_path is not None else default_lock_path(inventory_file)
    lock_file = requested_lock if requested_lock.is_absolute() else repository / requested_lock

    if lock_file.exists():
        lock = load_lock(lock_file)
        _verify_source_hashes(lock, inventory_file, root=repository)
    else:
        lock = build_lock(inventory_file, root=repository)
        write_lock(lock, lock_file)

    config_dir = (
        Path(frozen_dir)
        if frozen_dir is not None
        else default_frozen_dir(lock, root=repository)
    )
    if not config_dir.is_absolute():
        config_dir = repository / config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    for run in lock.runs:
        config = RunConfig.model_validate(run.config)
        if config_sha256(config) != run.config_sha256:
            raise ValueError(f"lock config digest mismatch for run {run.id!r}")
        path = config_dir / f"{run.id}.json"
        save_config(config, path)
        configs.append(path)

    manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else config_dir / "array-manifest.txt"
    )
    if not manifest.is_absolute():
        manifest = repository / manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # Relative paths make the manifest portable between submit hosts and checkout roots.
    manifest.write_text(
        "".join(f"{_display_path(path, repository)}\n" for path in configs),
        encoding="utf-8",
    )
    return lock, configs, manifest
