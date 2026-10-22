"""Draw a range of the coverage notebook's network injections into the cache.

``notebooks/shot_noise_coverage.py`` reads ``num_injections`` Poisson catalogs,
each recorded by the six networks, from
:func:`~astrogwb.simulators.spectra.network_spectra_ensemble`. Draw ``i`` is
the same whatever range it is drawn in, and each draw is one file per network
under ``<cache>/ensembles/<key>/<seed>-<i>.h5``, so several machines can each
draw a disjoint range and the files merge by copying. The notebook then finds
every draw cached.

The injection record below mirrors the notebook's configuration cell. A record
that differs gives other keys, and the notebook would redraw rather than read
these files: ``--keys`` prints the keys, which must match on every machine and
match the directories the notebook reads.

    uv run python scripts/draw_coverage_injections.py --keys
    uv run python scripts/draw_coverage_injections.py --start 0 --stop 250
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
from astrogwb.paper.plotting import DETECTOR_NETWORKS
from astrogwb.paper.posteriors import network_injection_metadata
from astrogwb.simulators.spectra import (
    BackgroundSpectralDensityMetadata,
    NetworkSpectralDensityMetadata,
    network_spectra_ensemble,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

# Keep in sync with the configuration cell of notebooks/shot_noise_coverage.py.
ENSEMBLE_SEED = 41
APPROXIMANT = "IMRPhenomXAS_NRTidalv3"
MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT = 0.35, 20.0
OBSERVATION_TIME = 1.0  # years
CHUNK_SIZE = 8192
SOURCE_CHUNK_SIZE = 65_536


def injection_records() -> list[NetworkSpectralDensityMetadata]:
    """The notebook's injection record of every network, in legend order."""
    registry = detector_registry(root=ROOT_DIR)
    injection = BackgroundSpectralDensityMetadata(
        count="poisson",
        num_events=None,
        observation_time=OBSERVATION_TIME,
        hyperparameters={**fiducials(root=ROOT_DIR)},
        waveform=waveform_metadata(
            root=ROOT_DIR,
            approximant=APPROXIMANT,
            frequency_resolution=0.125,
            turnover_frequency=100.0,
            dlogf=0.01266,
        ),
        population=population_metadata(
            root=ROOT_DIR,
            minimum_redshift=MINIMUM_REDSHIFT,
            maximum_redshift=MAXIMUM_REDSHIFT,
        ),
    )
    return [
        network_injection_metadata(injection, registry, name)
        for name, _ in DETECTOR_NETWORKS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=0, help="first draw index")
    parser.add_argument("--stop", type=int, help="one past the last draw index")
    parser.add_argument(
        "--keys", action="store_true", help="print each network's key and exit"
    )
    args = parser.parse_args()

    records = injection_records()
    directory = default_cache_dir() / "ensembles"
    if args.keys:
        for (name, _), record in zip(DETECTOR_NETWORKS, records, strict=True):
            print(f"{record.key()}  {name}")
        print(f"cache: {directory}")
        return
    if args.stop is None or not 0 <= args.start < args.stop:
        parser.error("give 0 <= --start < --stop")

    began = time.perf_counter()
    network_spectra_ensemble(
        records,
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
