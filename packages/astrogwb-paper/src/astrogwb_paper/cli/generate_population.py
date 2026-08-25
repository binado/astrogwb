"""Generate a single-component BNS population bank from one graph config.

Mixing MD and uniform-redshift components no longer happens here: each bank
is one component, generated once at one seed. Mixture composition happens
in-memory at catalog-load time -- see
:func:`astrogwb_paper.catalogs.compose_catalog`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gwmock_pop import GraphSimulator
from gwmock_pop.loaders.file_loader import (
    infer_population_file_format,
    write_population_catalogue,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a single-component BNS population bank from a graph config."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output population.",
    )
    return parser.parse_args(argv)


def simulate_population(
    config: Path,
    *,
    num_samples: int,
    seed: int,
) -> Mapping[str, Any]:
    """Draw a fresh single-component population."""
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    simulator = GraphSimulator.from_config_file(config, source_type="bns", seed=seed)
    return simulator.simulate(num_samples)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    output = args.output.expanduser()
    infer_population_file_format(output)
    if output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to replace existing population: {output}. "
            "Pass --force only for an intentional replacement."
        )

    population = simulate_population(
        args.config.expanduser(),
        num_samples=args.num_samples,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    write_population_catalogue(output, population)
    logger.info("Saved %d samples to %s", args.num_samples, output)


if __name__ == "__main__":
    main()
