"""Convert ensemble caches of per-stage background artifacts to realization directories.

Before the realization layout, each ensemble draw
``<cache>/ensembles/<key>/<seed>-<i>/`` was a background artifact:
``metadata.toml`` with ``[artifact]`` and ``[parent]`` tables, and
``data.h5``. Its manifest holds everything a realization manifest records --
the population and waveform metadata, the ``rng_key``, the detectors and the
realization -- and the seed and index come from the directory name, so a draw
converts in place without being redrawn: ``data.h5`` becomes
``background.h5``, whose datasets are unchanged, and ``metadata.toml`` is
rewritten. Every key is recomputed and checked against the recorded one first.

    uv run python scripts/migrate_ensemble_cache.py ~/.cache/astrogwb/ensembles --dry-run
    uv run python scripts/migrate_ensemble_cache.py ~/.cache/astrogwb/ensembles

A conversion interrupted between the two steps is finished by a rerun.
"""

from __future__ import annotations

import argparse
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from astrogwb.simulators import (
    DetectorConfiguration,
    PopulationMetadata,
    population_key,
    realization_key,
)
from astrogwb.simulators._io import (
    BACKGROUND_FILE,
    MANIFEST_FILE,
    write_realization_manifest,
)
from astrogwb.simulators.background import NORMALIZATION_VERSION, background_table
from astrogwb.simulators.waveforms import waveform_table
from astrogwb.waveform import WaveformMetadata

#: The data file of a per-stage artifact.
OLD_DATA_FILE = "data.h5"

_DRAW = re.compile(r"(?P<seed>\d+)-(?P<index>\d+)")


def converted_manifest(old: dict[str, Any], seed: int, index: int) -> dict[str, Any]:
    """The realization manifest of the background artifact manifest ``old``.

    Raises
    ------
    ValueError
        If ``old`` is not a background artifact, a recorded key does not match
        its recomputed value, or the rng key is not draw ``index`` of ``seed``.
    """
    artifact = old.get("artifact", {})
    if artifact.get("kind") != "background":
        raise ValueError(f"a {artifact.get('kind')!r} artifact, not a background")
    if old["normalization"].get("version") != NORMALIZATION_VERSION:
        raise ValueError("another pair-spectrum normalization version")
    parent = old["parent"]
    population = PopulationMetadata.model_validate(parent["population"]["metadata"])
    rng_key = [int(v) for v in parent["population"]["rng_key"]]
    expected = np.asarray(jax.random.key_data(realization_key(seed, index))).tolist()
    if rng_key != expected:
        raise ValueError(f"its rng key is not draw {index} of seed {seed}")
    pkey = population_key(population, rng_key)
    waveforms = waveform_table(
        WaveformMetadata.model_validate(parent["waveforms"]["metadata"]), pkey
    )
    background = background_table(
        waveforms["key"],
        DetectorConfiguration.model_validate(old["metadata"]["detectors"]),
    )
    for name, recorded, computed in (
        ("population", parent["population"]["key"], pkey),
        ("waveform", parent["waveforms"]["key"], waveforms["key"]),
        ("background", artifact["key"], background["key"]),
    ):
        if recorded != computed:
            raise ValueError(f"records {name} key {recorded}, recomputed {computed}")
    return {
        "realization": {
            "id": pkey,
            "seed": seed,
            "index": index,
            "rng_key": rng_key,
            **old["realization"],
        },
        "population": {"metadata": population.model_dump(mode="json"), "key": pkey},
        "waveforms": waveforms,
        "background": background,
        "execution": old.get("execution", {}),
    }


def migrate(directory: Path, *, dry_run: bool) -> bool:
    """Convert the draw at ``directory``; whether it was a per-stage artifact."""
    with (directory / MANIFEST_FILE).open("rb") as file:
        old = tomllib.load(file)
    if "artifact" not in old:
        return False
    match = _DRAW.fullmatch(directory.name)
    if match is None:
        raise ValueError(f"{directory}: not named <seed>-<index>")
    old_data, new_data = directory / OLD_DATA_FILE, directory / BACKGROUND_FILE
    if not (old_data.is_file() or new_data.is_file()):
        raise FileNotFoundError(
            f"{directory}: neither {OLD_DATA_FILE} nor {BACKGROUND_FILE}"
        )
    try:
        manifest = converted_manifest(old, int(match["seed"]), int(match["index"]))
    except ValueError as error:
        raise ValueError(f"{directory}: {error}") from error
    if not dry_run:
        if old_data.is_file():
            os.replace(old_data, new_data)
        write_realization_manifest(directory, manifest)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots", nargs="+", type=Path, help="cache directories to search for draws"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="check every draw, change nothing"
    )
    args = parser.parse_args()

    converted = kept = 0
    for root in args.roots:
        for manifest in sorted(root.rglob(MANIFEST_FILE)):
            if migrate(manifest.parent, dry_run=args.dry_run):
                converted += 1
            else:
                kept += 1
    verb = "would convert" if args.dry_run else "converted"
    print(f"{verb} {converted} draws; {kept} already in the realization layout")


if __name__ == "__main__":
    main()
