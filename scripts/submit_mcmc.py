#!/usr/bin/env python3
"""Submit a SLURM job array over MCMC configs in a directory or manifest."""

from __future__ import annotations

import argparse
import logging
import subprocess
import textwrap
from pathlib import Path
from uuid import uuid4

CONFIG_SUFFIXES = frozenset({".json", ".toml"})
logger = logging.getLogger(__name__)

BATCH_SCRIPT = textwrap.dedent(
    """\
    #!/bin/bash
    #SBATCH --job-name=asgwb-mcmc
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=8G
    #SBATCH --time=04:00:00
    #SBATCH --output=logs/mcmc_%A_%a.out
    #SBATCH --error=logs/mcmc_%A_%a.err
    #
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1

    set -euo pipefail

    mkdir -p logs

    SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
    if [[ "${CONFIG_MANIFEST}" != /* ]]; then
        CONFIG_MANIFEST="${SUBMIT_DIR}/${CONFIG_MANIFEST}"
    fi

    TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
    CONFIG=$(sed -n "$((TASK_ID + 1))p" "${CONFIG_MANIFEST}")
    if [[ -z "${CONFIG}" ]]; then
        echo "Error: No config for array task ${TASK_ID} in ${CONFIG_MANIFEST}" >&2
        exit 1
    fi
    if [[ "${CONFIG}" != /* ]]; then
        CONFIG="${SUBMIT_DIR}/${CONFIG}"
    fi
    if [[ ! -f "${CONFIG}" ]]; then
        echo "Error: Config file '${CONFIG}' not found." >&2
        exit 1
    fi

    cd "${SUBMIT_DIR}"

    # Pin CPU threads to the allocation so XLA/OMP do not oversubscribe the node.
    # (run_mcmc.py also honors [runtime] cpu_threads; either is sufficient.)
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export INPUT="*"
    export OUTPUT="*"
    export JAX_PLATFORMS="cuda"

    echo "Task ${TASK_ID}: running ${CONFIG}"

    if command -v job-nanny >/dev/null 2>&1; then
        job-nanny uv run --extra mcmc python scripts/run_mcmc.py --config "${CONFIG}"
    else
        uv run --extra mcmc python scripts/run_mcmc.py --config "${CONFIG}"
    fi
    """
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        help="directory containing one TOML or JSON config per array task",
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "-i",
        "--input",
        type=Path,
        help="directory containing one TOML or JSON config per array task",
    )
    inputs.add_argument(
        "--manifest",
        type=Path,
        help="non-empty file with one existing TOML or JSON config per line",
    )
    parser.add_argument(
        "--campaign-lock",
        type=Path,
        help="immutable campaign lock to validate before submitting",
    )
    parser.add_argument(
        "--bypass-locks",
        action="store_true",
        help="skip campaign-lock validation (for debugging or known legacy data)",
    )
    args = parser.parse_args(argv)

    if args.input_dir is not None and args.input is not None:
        parser.error("positional config directory and --input are mutually exclusive")
    if args.input_dir is not None and args.manifest is not None:
        parser.error(
            "positional config directory and --manifest are mutually exclusive"
        )
    if args.input_dir is None and args.input is None and args.manifest is None:
        parser.error("missing config directory or manifest")
    return args


def validate_manifest(manifest: Path) -> list[str]:
    if not manifest.is_file():
        raise ValueError(f"Manifest '{manifest}' not found.")

    configs = manifest.read_text(encoding="utf-8").splitlines()
    if not configs:
        raise ValueError(f"Manifest '{manifest}' is empty.")

    for config in configs:
        if not config:
            raise ValueError(f"Manifest '{manifest}' contains an empty line.")
        path = Path(config)
        if path.suffix not in CONFIG_SUFFIXES:
            raise ValueError(f"Manifest entry '{config}' is not a TOML or JSON config.")
        if not path.is_file():
            raise ValueError(f"Manifest config '{config}' not found.")
    return configs


def configs_in_directory(input_dir: Path) -> list[str]:
    if not input_dir.is_dir():
        raise ValueError(f"Config directory '{input_dir}' not found.")

    configs = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix in CONFIG_SUFFIXES
    )
    if not configs:
        raise ValueError(f"No *.toml or *.json configs found in '{input_dir}'.")
    return [str(path) for path in configs]


def write_manifest(configs: list[str]) -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    manifest = logs_dir / f"mcmc_manifest_{uuid4().hex}.txt"
    manifest.write_text("\n".join(configs) + "\n", encoding="utf-8")
    return manifest


def validate_campaign_lock(lock_path: Path, configs: list[str]) -> None:
    """Ensure a complete frozen campaign and its catalogs match ``lock_path``."""
    from astrogwb.sampling.campaign import config_sha256, file_sha256, load_lock
    from astrogwb.sampling.config import RunConfig, build_run_config, load_config

    lock = load_lock(lock_path)
    missing_digests = [run.id for run in lock.runs if run.catalog_sha256 is None]
    if missing_digests:
        raise ValueError(
            f"Campaign lock '{lock_path}' does not pin catalog SHA-256 values for "
            f"run(s): {', '.join(missing_digests)}. Use --bypass-locks only for "
            "intentional legacy submissions."
        )

    expected = {run.config_sha256: run for run in lock.runs}
    if len(expected) != len(lock.runs):
        raise ValueError(f"Campaign lock '{lock_path}' has duplicate config digests.")

    submitted: dict[str, RunConfig] = {}
    submitted_paths: dict[str, str] = {}
    for config_path in configs:
        config = build_run_config(load_config(Path(config_path)))
        digest = config_sha256(config)
        if digest in submitted:
            run_id = expected[digest].id if digest in expected else config_path
            raise ValueError(
                f"Campaign manifest submits config {run_id!r} more than once."
            )
        submitted[digest] = config
        submitted_paths[digest] = config_path

    if set(submitted) != set(expected):
        missing = [
            run.id for digest, run in expected.items() if digest not in submitted
        ]
        unexpected = [
            config_path
            for digest, config_path in submitted_paths.items()
            if digest not in expected
        ]
        details: list[str] = []
        if missing:
            details.append(f"missing locked run(s): {', '.join(missing)}")
        if unexpected:
            details.append(f"unlocked config(s): {', '.join(unexpected)}")
        raise ValueError(
            f"Campaign manifest must contain every config from '{lock_path}' exactly "
            f"once ({'; '.join(details)})."
        )

    catalogs: dict[Path, str] = {}
    for digest, config in submitted.items():
        expected_digest = expected[digest].catalog_sha256
        assert expected_digest is not None
        catalog_path = config.catalog.path
        if not catalog_path.is_absolute():
            catalog_path = Path.cwd() / catalog_path
        catalog_path = catalog_path.resolve()
        previous_digest = catalogs.setdefault(catalog_path, expected_digest)
        if previous_digest != expected_digest:
            raise ValueError(
                f"Campaign lock '{lock_path}' assigns conflicting SHA-256 values to "
                f"catalog '{catalog_path}'."
            )

    for catalog_path, expected_digest in catalogs.items():
        if not catalog_path.is_file():
            raise ValueError(f"Locked catalog '{catalog_path}' not found.")
        actual_digest = file_sha256(catalog_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Locked catalog '{catalog_path}' SHA-256 mismatch: expected "
                f"{expected_digest}, got {actual_digest}."
            )


def submit_array(manifest: Path, config_count: int) -> subprocess.CompletedProcess[str]:
    # SLURM opens the --output/--error files before the batch script body runs,
    # so logs/ must exist at submission time; the in-job mkdir is too late.
    Path("logs").mkdir(exist_ok=True)
    return subprocess.run(
        [
            "sbatch",
            f"--array=0-{config_count - 1}",
            f"--export=ALL,CONFIG_MANIFEST={manifest}",
        ],
        input=BATCH_SCRIPT,
        text=True,
        capture_output=True,
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        if args.manifest is not None:
            manifest = args.manifest
            configs = validate_manifest(manifest)
        else:
            input_dir = args.input if args.input is not None else args.input_dir
            assert input_dir is not None
            configs = configs_in_directory(input_dir)
            manifest = write_manifest(configs)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    if args.bypass_locks:
        logger.warning("bypassing campaign-lock validation")
    elif args.campaign_lock is not None:
        try:
            validate_campaign_lock(args.campaign_lock, configs)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Error: {error}") from error

    logger.info("submitting %d MCMC jobs (array 0-%d)", len(configs), len(configs) - 1)
    logger.info("manifest: %s", manifest)
    for index, config in enumerate(configs):
        logger.info("  [%d] %s", index, config)

    try:
        result = submit_array(manifest, len(configs))
    except FileNotFoundError as error:
        raise SystemExit("Error: 'sbatch' was not found on PATH.") from error
    except subprocess.CalledProcessError as error:
        if error.stdout:
            logger.info("%s", error.stdout.rstrip())
        if error.stderr:
            logger.error("%s", error.stderr.rstrip())
        raise SystemExit(error.returncode) from error

    if result.stdout:
        logger.info("%s", result.stdout.rstrip())
    if result.stderr:
        logger.error("%s", result.stderr.rstrip())


if __name__ == "__main__":
    main()
