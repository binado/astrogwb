"""Generate one waveform catalog from its resolved metadata.

The workflow's ``waveform_catalog`` rule is the caller: every run's
``[analysis.injection]`` and ``[analysis.proposal]`` are validated into
:class:`~astrogwb.simulators.polarization_power.CatalogMetadata` records when the DAG is built, and
each distinct ``(record, seed)`` becomes one job writing
``outputs/catalogs/polarization_power-<key>-<digest>.h5``. This script is handed
that record as JSON, with the seed, and builds it with the cached
:func:`~astrogwb.simulators.polarization_power.polarization_power` node, which
writes it atomically.

The output must be ``polarization_power.path`` of the record and seed in its own
directory. The workflow names the file and the record separately, so this is
where a mismatch between the two -- which would file one draw under another's
address -- is refused. An output that already exists is a cache hit: it is
checked against the record and left alone, unless ``--force`` asks for it
to be drawn again.

Outside the workflow, the node is the whole thing, and
needs no script.

Usage::

    uv run --extra paper python scripts/generate_catalog.py \\
        --request "$(cat request.json)" --seed 41 \\
        --output outputs/catalogs/polarization_power-<key>-<digest>.h5
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from astrogwb.paper.config.catalogs import check_population_model
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    PolarizationPowerCatalog,
    polarization_power,
)

logger = logging.getLogger(__name__)


def _request(raw: str) -> CatalogMetadata:
    """Parse and validate the catalog metadata off argv."""
    try:
        return CatalogMetadata.model_validate_json(raw)
    except ValidationError as error:
        raise argparse.ArgumentTypeError(f"invalid catalog request: {error}") from None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a BNS population from its registered NumPyro model, generate "
            "frequency-domain waveforms, and persist the polarization power as "
            "an astrogwb_catalog HDF5 file named by its metadata's key."
        )
    )
    parser.add_argument(
        "--request",
        required=True,
        type=_request,
        metavar="JSON",
        help="The resolved CatalogMetadata, as JSON.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="The seed that picks the realization (non-negative integer).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .h5; its stem must be polarization_power.path's.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Draw an existing output catalog again instead of reusing it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    metadata: CatalogMetadata = args.request
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    inputs = {"seed": np.uint64(args.seed)}
    output_path = args.output.expanduser().resolve()
    cache_dir = output_path.parent
    if output_path != polarization_power.path(inputs, metadata, cache_dir).resolve():
        raise ValueError(
            f"output {output_path.name} is not named by the metadata's key "
            f"{metadata.key()} and seed {args.seed}"
        )

    population = metadata.population
    check_population_model(population, label=f"catalog {metadata.key()} population")
    if args.force:
        output_path.unlink(missing_ok=True)
    outputs = polarization_power(inputs, metadata, cache_dir=cache_dir)
    catalog = PolarizationPowerCatalog.from_arrays(outputs, metadata)

    logger.info(
        "Catalog %s: %d events, %d frequencies (%.2f-%.2f Hz), approximant=%s",
        metadata.key(),
        catalog.num_samples,
        catalog.frequencies.size,
        catalog.frequencies[0].item(),
        catalog.frequencies[-1].item(),
        metadata.waveform.approximant,
    )
    logger.info("Catalog at %s", output_path)


if __name__ == "__main__":
    main()
