"""Draw one reference catalog of the importance-weighted likelihood to a file.

The posterior notebooks rescale a reference-redshift catalog: intrinsic draws
at the fiducials, each generated once at the window's lower edge (or at
``--reference-redshift``) on a log-spaced reference grid. This draws it at
``batch_keys(seed, 1)[0]`` from the shared paper config, with the variations
the notebooks use as options, and writes it to ``--output``. Notebooks read
the file with ``astrogwb.utils.io.load(path, CatalogMetadata)`` and never draw.

    uv run python scripts/draw_reference_catalog.py \\
        --output outputs/catalogs/reference-fiducial.h5
    uv run python scripts/draw_reference_catalog.py --reference-redshift 2.2565e-4 \\
        --output outputs/catalogs/reference-scan.h5

``workflow/coverage_draws.smk`` runs it for every catalog the notebooks read.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from astrogwb.paper.runtime import configure_runtime

# Backend configuration precedes waveform construction and array creation.
configure_runtime(num_chains=1)

from astrogwb.gwb.catalog import CatalogMetadata
from astrogwb.gwb.importance import reference_catalog
from astrogwb.paper.config import fiducials, population_metadata, waveform_metadata
from astrogwb.simulators import batch_keys
from astrogwb.utils.io import write

ROOT_DIR = Path(__file__).resolve().parents[1]


def catalog_metadata(args: argparse.Namespace) -> CatalogMetadata:
    """The catalog's record: the shared config with the options' variations."""
    waveform = {
        "frequency_spacing": "log",
        "minimum_frequency": args.minimum_frequency,
        "maximum_frequency": args.maximum_frequency,
        "frequency_resolution": args.frequency_resolution,
        "turnover_frequency": None,
    }
    if args.approximant is not None:
        waveform["approximant"] = args.approximant
    redshifts = {
        name: value
        for name, value in (
            ("minimum_redshift", args.minimum_redshift),
            ("maximum_redshift", args.maximum_redshift),
        )
        if value is not None
    }
    hyperparameters = fiducials(ROOT_DIR)
    return CatalogMetadata(
        waveform=waveform_metadata(ROOT_DIR, **waveform),
        population=population_metadata(ROOT_DIR, **redshifts),
        fiducials=hyperparameters,
        num_samples=args.num_samples,
        sampling=args.sampling,
        reference_redshift=args.reference_redshift,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="HDF5 file to write")
    parser.add_argument("--seed", type=int, default=42, help="(default: %(default)s)")
    parser.add_argument(
        "--num-samples", type=int, default=2**14, help="(default: %(default)s)"
    )
    parser.add_argument(
        "--sampling",
        choices=("random", "sobol"),
        default="sobol",
        help="(default: %(default)s)",
    )
    parser.add_argument(
        "--approximant", help="waveform approximant (default: the shared one)"
    )
    parser.add_argument(
        "--minimum-frequency", type=float, default=2.0, help="(default: %(default)s)"
    )
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        default=4000.0,
        help="past the detector-frame merger (default: %(default)s)",
    )
    parser.add_argument(
        "--frequency-resolution",
        type=float,
        default=0.02,
        help="relative width of the log bins (default: %(default)s)",
    )
    parser.add_argument(
        "--minimum-redshift", type=float, help="(default: the shared window)"
    )
    parser.add_argument(
        "--maximum-redshift", type=float, help="(default: the shared window)"
    )
    parser.add_argument(
        "--reference-redshift",
        type=float,
        help="redshift every draw is generated at (default: the window's lower edge)",
    )
    parser.add_argument("--chunk-size", type=int, help="waveforms per batch; cost only")
    args = parser.parse_args()

    metadata = catalog_metadata(args)
    began = time.perf_counter()
    data = reference_catalog(
        metadata, batch_keys(args.seed, 1)[0], chunk_size=args.chunk_size
    )
    write(args.output, data, metadata, seed=args.seed)
    elapsed = time.perf_counter() - began
    print(
        f"reference catalog {metadata.key()} at seed {args.seed}: "
        f"{metadata.num_samples} draws -> {args.output} ({elapsed:.0f} s)"
    )


if __name__ == "__main__":
    main()
