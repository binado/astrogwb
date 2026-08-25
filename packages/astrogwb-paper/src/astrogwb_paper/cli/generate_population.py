"""Generate a BNS population from an MD/uniform-redshift mixture."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gwmock_pop import GraphSimulator, MixtureSimulator
from gwmock_pop.loaders.file_loader import (
    infer_population_file_format,
    write_population_catalogue,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a BNS population from Madau-Dickinson and "
            "uniform-redshift graph configurations."
        )
    )
    parser.add_argument("--md-config", type=Path, required=True)
    parser.add_argument("--uniform-redshift-config", type=Path, required=True)
    parser.add_argument(
        "--uniform-mixing-fraction",
        type=float,
        required=True,
        help="Uniform-redshift mixture probability, inclusive in [0, 1].",
    )
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
    md_config: Path,
    uniform_redshift_config: Path,
    *,
    uniform_mixing_fraction: float,
    num_samples: int,
    seed: int,
) -> Mapping[str, Any]:
    """Draw a fresh population from the requested redshift mixture."""
    epsilon = float(uniform_mixing_fraction)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("uniform_mixing_fraction must satisfy 0 <= epsilon <= 1")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    if epsilon == 0.0:
        simulator = GraphSimulator.from_config_file(
            md_config, source_type="bns", seed=seed
        )
    elif epsilon == 1.0:
        simulator = GraphSimulator.from_config_file(
            uniform_redshift_config, source_type="bns", seed=seed
        )
    else:
        # MixtureSimulator seeds component calls, but GraphSimulator draws
        # from its construction-time RNG, so the components must be seeded
        # here for the mixture to be reproducible. All three seeds must
        # differ: RNGManager starts from jax.random.key(seed), so a shared
        # seed means identical key streams across simulators.
        md = GraphSimulator.from_config_file(
            md_config, source_type="bns", seed=seed + 1
        )
        uniform = GraphSimulator.from_config_file(
            uniform_redshift_config, source_type="bns", seed=seed + 2
        )
        simulator = MixtureSimulator(
            [md, uniform],
            [1.0 - epsilon, epsilon],
            seed=seed,
        )
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
        args.md_config.expanduser(),
        args.uniform_redshift_config.expanduser(),
        uniform_mixing_fraction=args.uniform_mixing_fraction,
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
