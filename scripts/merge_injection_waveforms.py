"""Merge multiple per-batch HDF5 waveform files into a single file.

Usage:
    python scripts/merge_injection_waveforms.py \
        "output_dir/waveforms_batch_*.h5" \
        output_dir/merged.h5
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from glob import glob
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)


def _natural_sort_key(path: str) -> list[int | str]:
    """Sort key that orders embedded numbers numerically."""
    parts: list[int | str] = []
    for part in re.split(r"(\d+)", Path(path).stem):
        parts.append(int(part) if part.isdigit() else part)
    return parts


def _get_git_revision() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def merge_hdf5_files(
    input_files: list[Path],
    output_file: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Concatenate per-batch HDF5 waveform files along the injection axis.

    Each input file must have the columnar layout produced by
    ``generate_injection_waveforms.py``::

        polarizations/{plus,cross}  — shape (N_i, F), complex128
        parameters/{key}            — shape (N_i,)
    """
    if not input_files:
        logger.error("No input files provided")
        sys.exit(1)

    # --- Scan pass: compute total N and validate schema consistency ---
    total_n = 0
    freq_bins: int | None = None
    param_keys: list[str] | None = None

    for path in input_files:
        with h5py.File(path, "r") as hf:
            n, f = hf["polarizations/plus"].shape
            total_n += n

            file_param_keys = sorted(hf["parameters"].keys())
            if freq_bins is None:
                freq_bins = f
                param_keys = file_param_keys
            else:
                if f != freq_bins:
                    logger.error(
                        "Frequency bins mismatch: expected %d, got %d in %s",
                        freq_bins,
                        f,
                        path,
                    )
                    sys.exit(1)
                if file_param_keys != param_keys:
                    logger.error(
                        "Parameter keys mismatch in %s: expected %s, got %s",
                        path,
                        param_keys,
                        file_param_keys,
                    )
                    sys.exit(1)

    assert freq_bins is not None and param_keys is not None

    logger.info(
        "Merging %d files (%d total injections, %d frequency bins)",
        len(input_files),
        total_n,
        freq_bins,
    )

    # --- Pre-allocate output ---
    with h5py.File(output_file, "w") as out:
        pol_group = out.create_group("polarizations")
        ds_plus = pol_group.create_dataset(
            "plus",
            shape=(total_n, freq_bins),
            dtype="complex128",
            compression="gzip",
        )
        ds_cross = pol_group.create_dataset(
            "cross",
            shape=(total_n, freq_bins),
            dtype="complex128",
            compression="gzip",
        )

        par_group = out.create_group("parameters")
        ds_params = {
            key: par_group.create_dataset(
                key, shape=(total_n,), dtype="float64", compression="gzip"
            )
            for key in param_keys
        }

        # --- Propagate source metadata from first input file ---
        with h5py.File(input_files[0], "r") as first_hf:
            if "args" in first_hf.attrs:
                source_args = json.loads(first_hf.attrs["args"])
                source_args.pop("offset", None)
                source_args.pop("batch", None)
                out.attrs["source_args"] = json.dumps(source_args, default=str)
            if "git_revision" in first_hf.attrs:
                out.attrs["source_git_revision"] = first_hf.attrs["git_revision"]

        if metadata:
            for k, v in metadata.items():
                out.attrs[k] = v

        # --- Fill pass ---
        offset = 0
        for path in input_files:
            with h5py.File(path, "r") as hf:
                n = hf["polarizations/plus"].shape[0]
                sl = slice(offset, offset + n)
                ds_plus[sl] = hf["polarizations/plus"][:]
                ds_cross[sl] = hf["polarizations/cross"][:]
                for key in param_keys:
                    ds_params[key][sl] = hf[f"parameters/{key}"][:]
                offset += n

    logger.info("Wrote merged file to %s", output_file)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Merge per-batch HDF5 waveform files into a single file"
    )
    parser.add_argument(
        "input_pattern",
        help="Glob pattern for input HDF5 files (quote to avoid shell expansion)",
    )
    parser.add_argument("output_file", type=Path, help="Path to the merged output file")
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort input files by name (default: natural sort)",
    )
    args = parser.parse_args()

    input_files = glob(args.input_pattern)
    if not input_files:
        logger.error("No files matched pattern: %s", args.input_pattern)
        sys.exit(1)

    if not args.no_sort:
        input_files.sort(key=_natural_sort_key)

    metadata: dict[str, str] = {"command": json.dumps(sys.argv)}
    git_rev = _get_git_revision()
    if git_rev is not None:
        metadata["git_revision"] = git_rev

    merge_hdf5_files(
        [Path(f) for f in input_files], args.output_file, metadata=metadata
    )


if __name__ == "__main__":
    main()
