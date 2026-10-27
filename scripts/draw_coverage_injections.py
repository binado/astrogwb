"""Draw a range of the coverage notebook's network injections into the cache.

``notebooks/shot_noise_coverage.py`` reads ``num_injections`` Poisson catalogs,
each recorded by the six networks, from
:func:`~astrogwb.paper.ensemble.network_spectra_ensemble`. Draw ``i`` is the
same whatever range it is drawn in, and each draw is one realization
directory, its manifest and one background over the union of the networks'
detectors, under ``<cache>/ensembles/<key>/<seed>-<i>/``, so several machines
can each draw a disjoint range and the directories merge by copying. The
notebook then finds every draw cached. Caches drawn before the realization
layout are converted in place by ``scripts/migrate_ensemble_cache.py``.

The injection record below mirrors the notebook's configuration cell. A record
that differs gives another key, and the notebook would redraw rather than read
these files: ``--keys`` prints the key, which must match on every machine and
match the directory the notebook reads.

    uv run python scripts/draw_coverage_injections.py --keys
    uv run python scripts/draw_coverage_injections.py --start 0 --stop 250

``workflow/coverage_draws.smk`` runs this script as SLURM jobs, each a small
range of draws; see ``docs/coverage-draws-on-slurm.md``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from astrogwb.paper.runtime import configure_runtime

# Backend configuration precedes waveform construction and array creation.
configure_runtime(num_chains=1)

from astrogwb.paper.cache import default_cache_dir
from astrogwb.paper.config import (
    detector_registry,
    fiducials,
    population_metadata,
    waveform_metadata,
)
from astrogwb.paper.ensemble import (
    EnsembleRecord,
    network_configuration,
    network_spectra_ensemble,
)
from astrogwb.paper.plotting import DETECTOR_NETWORKS
from astrogwb.simulators import PopulationMetadata

ROOT_DIR = Path(__file__).resolve().parents[1]

# Keep in sync with the configuration cell of notebooks/shot_noise_coverage.py.
ENSEMBLE_SEED = 41
APPROXIMANT = "IMRPhenomXAS_NRTidalv3"
MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT = 0.35, 20.0
OBSERVATION_TIME = 1.0  # years
CHUNK_SIZE = 8192
SOURCE_CHUNK_SIZE = 65_536


def injection_record() -> EnsembleRecord:
    """The notebook's injection record, its networks in legend order."""
    registry = detector_registry(root=ROOT_DIR)
    population = PopulationMetadata(
        observation_time=OBSERVATION_TIME,
        hyperparameters={**fiducials(root=ROOT_DIR)},
        population=population_metadata(
            root=ROOT_DIR,
            minimum_redshift=MINIMUM_REDSHIFT,
            maximum_redshift=MAXIMUM_REDSHIFT,
        ),
    )
    waveform = waveform_metadata(
        root=ROOT_DIR,
        approximant=APPROXIMANT,
        frequency_resolution=0.125,
        turnover_frequency=100.0,
        dlogf=0.01266,
    )
    return EnsembleRecord(
        population=population,
        waveform=waveform,
        networks={
            name: network_configuration(registry, name) for name, _ in DETECTOR_NETWORKS
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=0, help="first draw index")
    parser.add_argument("--stop", type=int, help="one past the last draw index")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="cache root; draws land under <cache-dir>/ensembles (default: %(default)s)",
    )
    parser.add_argument(
        "--keys", action="store_true", help="print the ensemble key and exit"
    )
    args = parser.parse_args()

    record = injection_record()
    directory = args.cache_dir / "ensembles"
    if args.keys:
        print(record.key())
        print(f"cache: {directory}")
        return
    if args.stop is None or not 0 <= args.start < args.stop:
        parser.error("give 0 <= --start < --stop")

    began = time.perf_counter()
    network_spectra_ensemble(
        record,
        ENSEMBLE_SEED,
        args.stop,
        directory,
        start=args.start,
        chunk_size=CHUNK_SIZE,
        source_chunk_size=SOURCE_CHUNK_SIZE,
    )
    elapsed = time.perf_counter() - began
    print(
        f"draws {args.start}..{args.stop - 1} of seed {ENSEMBLE_SEED} in "
        f"{directory} ({elapsed / 60:.1f} min)"
    )


if __name__ == "__main__":
    main()
