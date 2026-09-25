"""Generate one waveform catalog from its resolved request.

The workflow's ``waveform_catalog`` rule is the caller: every run's
``[analysis.catalog]`` roles are resolved into
:class:`~astrogwb.metadata.CatalogRequest` records when the DAG is built, and
each distinct request becomes one job writing ``outputs/catalogs/<key>.h5``.
This script is handed that request as JSON, generates it with
:func:`astrogwb.catalog.generate`, and writes it atomically.

The output's stem must be the request's key. The workflow names the file and
the request separately, so this is where a mismatch between the two -- which
would file one draw under another's address -- is refused.

Outside the workflow, :func:`astrogwb.catalog.load_or_generate` is the same
generator behind a cache lookup, and needs no script.

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

from astrogwb.catalog import generate, save_atomically
from astrogwb.metadata import CatalogRequest
from astrogwb.paper.config.catalogs import check_population_model

logger = logging.getLogger(__name__)


def _request(raw: str) -> CatalogRequest:
    """Parse and validate the request off argv."""
    try:
        return CatalogRequest.model_validate_json(raw)
    except ValidationError as error:
        raise argparse.ArgumentTypeError(f"invalid catalog request: {error}") from None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a BNS population from its registered NumPyro model, generate "
            "frequency-domain waveforms, and persist the polarization power as "
            "an astrogwb_catalog HDF5 file named by the request's key."
        )
    )
    parser.add_argument(
        "--request",
        required=True,
        type=_request,
        metavar="JSON",
        help="The resolved CatalogRequest, as JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .h5; its stem must be the request's key.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output catalog.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    request: CatalogRequest = args.request
    output_path = args.output.expanduser().resolve()
    if output_path.stem != request.key():
        raise ValueError(
            f"output {output_path.name} is not named by the request's key "
            f"{request.key()}"
        )
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"refusing to replace existing catalog: {output_path}. "
            "Pass --force only for an intentional replacement."
        )

    population = request.metadata.population
    check_population_model(
        population.model_name,
        label=f"catalog {request.key()} population.model_name",
        kwargs=population.model_kwargs,
    )
    catalog = generate(request)
    save_atomically(catalog, output_path)

    logger.info(
        "Saved catalog %s: %d events, %d frequencies (%.2f-%.2f Hz), approximant=%s",
        request.key(),
        catalog.num_samples,
        catalog.frequencies.size,
        catalog.frequencies[0].item(),
        catalog.frequencies[-1].item(),
        request.metadata.waveform.approximant,
    )
    logger.info("Output written to %s", output_path)


if __name__ == "__main__":
    main()
