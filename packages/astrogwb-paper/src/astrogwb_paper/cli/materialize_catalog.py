"""Materialize a named catalog composition to a waveform_catalog HDF5 file.

Compositions are cheap and in-memory by design (see
:func:`astrogwb_paper.catalogs.compose_catalog`) and the workflow never
writes one to disk. This CLI exists purely for inspection/debugging -- e.g.
pointing an ad-hoc tool at a single materialized file -- and is not a
Snakemake rule.

Usage::

    uv run astrogwb-materialize-catalog \
        --catalog bns-n16384-eps=0.1-df1 \
        --md-bank outputs/banks/md-imrphenom-s42.h5 \
        --uniform-bank outputs/banks/uniform-imrphenom-s51.h5 \
        --output /tmp/bns-n16384-eps=0.1-df1.h5
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose a named catalog from its bank file(s) and write it to disk "
            "in waveform_catalog format."
        )
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Composition name from inputs/catalogs.yaml.",
    )
    parser.add_argument(
        "--md-bank",
        type=Path,
        required=True,
        help="Path to the composition's MD bank file.",
    )
    parser.add_argument(
        "--uniform-bank",
        type=Path,
        default=None,
        help=(
            "Path to the composition's uniform-redshift bank file; required "
            "only when the composition mixes one in (uniform_mixing_fraction > 0)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output catalog.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from astrogwb.waveform import save_catalog

    from astrogwb_paper.catalogs import compose_catalog
    from astrogwb_paper.config.catalogs import catalog_recipe

    composition = catalog_recipe(args.catalog)
    output = args.output.expanduser()
    if output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to replace existing catalog: {output}. "
            "Pass --force only for an intentional replacement."
        )

    uniform_bank = args.uniform_bank.expanduser() if args.uniform_bank else None
    catalog = compose_catalog(args.md_bank.expanduser(), uniform_bank, composition)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_catalog(output, catalog)
    logger.info(
        "Materialized composition %r (%d samples) to %s",
        args.catalog,
        catalog.sizes["sample"],
        output,
    )


if __name__ == "__main__":
    main()
