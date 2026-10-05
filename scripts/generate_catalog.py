"""Generate one waveform catalog from its resolved metadata.

The workflow's ``waveform_catalog`` rule is the caller: every run's
``[analysis.injection]`` and ``[analysis.proposal]`` are validated into
:class:`~astrogwb.simulators.polarization_power.CatalogMetadata` records when the DAG is built, and
each distinct record becomes one job writing ``outputs/catalogs/<key>.h5``.
This script is handed that record as JSON and builds it with
:func:`astrogwb.simulators.core.simulate` and a
:class:`~astrogwb.simulators.polarization_power.CatalogGenerator`, which writes it atomically.

The output must be the record's :func:`~astrogwb.simulators.core.artifact_path` in
its own directory. The workflow names the file and the record separately, so
this is where a mismatch between the two -- which would file one draw under
another's address -- is refused. An output that already exists is a cache hit:
it is checked against the record and left alone, unless ``--force`` asks for it
to be drawn again.

Outside the workflow, :func:`astrogwb.simulators.core.simulate` is the whole thing, and
needs no script.

Usage::

    uv run --extra paper python scripts/generate_catalog.py \\
        --request "$(cat request.json)" \\
        --output outputs/catalogs/<key>.h5
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from astrogwb.paper.config.catalogs import check_population_model
from astrogwb.simulators.core import artifact_path, simulate
from astrogwb.simulators.polarization_power import CatalogGenerator, CatalogMetadata

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
        "--output",
        type=Path,
        required=True,
        help="Destination .h5; its stem must be the metadata's key.",
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
    output_path = args.output.expanduser().resolve()
    cache_dir = output_path.parent
    if output_path != artifact_path(metadata, cache_dir):
        raise ValueError(
            f"output {output_path.name} is not named by the metadata's key "
            f"{metadata.key()}"
        )

    population = metadata.population
    check_population_model(
        population.model_name,
        label=f"catalog {metadata.key()} population.model_name",
        kwargs=population.model_kwargs,
    )
    if args.force:
        output_path.unlink(missing_ok=True)
    catalog = simulate(metadata, CatalogGenerator(), cache_dir)

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
