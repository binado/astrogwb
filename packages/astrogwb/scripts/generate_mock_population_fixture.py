# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gwmock-pop==0.11.4",
#   "numpy>=2.5.1",
#   "pyyaml>=6.0.3",
# ]
# ///
"""Generate a committed mock BNS population fixture.

Draws sources from an explicitly supplied population graph and writes the
selected columns to an explicitly supplied CSV. The core test suite turns the
committed draw into a real ``WaveformCatalog`` without touching a Ripple
backend or a persisted bank.

The repository's canonical invocation is::

    just generate-mock-population-fixture

It uses the self-contained population graph committed beside the CSV under
``tests/fixtures``. Direct invocations must pass the population path, output
path, sample count, and seed explicitly.

``gwmock-pop`` is pinned *exactly* here even though ``astrogwb`` depends on it
with a floor (``>=0.11.4``): the graph YAML's own header warns that
``GraphSimulator`` draws RNG keys in topological order and breaks ties by
declaration order, so a patch release that reorders anything would resample the
population. The fixture has to be reproducible from this script alone.

The CSV is committed rather than simulated in ``conftest.py`` so the isolated
core tests do not regenerate their oracle or depend on generator-version RNG
details. The population graph is likewise a frozen core test input rather than
a live reference to another workspace package.

Only regenerate this when intentionally updating the reference (for example, to
bump the pinned ``gwmock-pop``); astrogwb's own tests treat it as a fixed
external oracle, not a target to chase.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml
from gwmock_pop import GraphSimulator

#: Columns written, in order. ``luminosity_distance`` is deliberately *not*
#: among them: the test factory recomputes it from
#: ``compute_merger_rate_distance_and_logprob`` at the fiducials, so the
#: "catalog is both injection and proposal" identity gives ``log w == 0`` by
#: construction rather than to CSV round-trip precision.
COLUMNS = (
    "redshift",
    "source_frame_mass_1",
    "source_frame_mass_2",
    "inclination",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw the committed mock BNS population fixture from a gwmock-pop "
            "graph config."
        ),
    )
    parser.add_argument(
        "--population",
        type=pathlib.Path,
        required=True,
        help="path to the population graph YAML",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=True,
        help="path to write the fixture CSV to",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        required=True,
        help="number of sources to draw",
    )
    parser.add_argument("--seed", type=int, required=True, help="GraphSimulator seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.population.is_file():
        raise SystemExit(f"population graph not found: {args.population}")

    raw_config = yaml.safe_load(args.population.read_text(encoding="utf-8"))
    population_name = raw_config.get("name") if isinstance(raw_config, dict) else None
    if not isinstance(population_name, str) or not population_name:
        raise SystemExit(
            f"{args.population}: population graph must declare a non-empty name"
        )

    simulator = GraphSimulator.from_config_file(
        args.population, source_type="bns", seed=args.seed
    )
    population = dict(simulator.simulate(args.num_samples))
    missing = [name for name in COLUMNS if name not in population]
    if missing:
        raise SystemExit(
            f"{args.population}: graph does not produce {', '.join(missing)}"
        )

    table = np.column_stack(
        [np.asarray(population[name], dtype=np.float64) for name in COLUMNS]
    )
    provenance = (
        f"population={population_name} seed={args.seed} "
        f"num_samples={args.num_samples} gwmock-pop=0.11.4"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The provenance line is written first and the column names second, so
    # readers pass ``skip_header=1``: ``np.genfromtxt(names=True)`` takes the
    # first *physical* line as the name row and would otherwise parse the
    # comment as column names.
    np.savetxt(
        args.output,
        table,
        fmt="%.17g",
        delimiter=",",
        header=f"# {provenance}\n" + ",".join(COLUMNS),
        comments="",
    )
    print(f"Wrote {args.num_samples} sources to {args.output}")
    print(f"  {provenance}")


if __name__ == "__main__":
    main()
