# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gwmock-pop==0.11.4",
#   "numpy>=2.5.1",
# ]
# ///
"""Generate the committed mock BNS population fixture.

Draws the first ``N_FIXTURE`` sources of the paper's pinned population graph
and writes them to ``tests/fixtures/mock_bns_population.csv``. The core test
suite turns that draw into a real ``WaveformCatalog`` (see the
``mock_catalog_factory`` fixture in ``tests/conftest.py``) without ever
touching a Ripple backend or a persisted bank.

Run it in the isolated, throwaway environment ``uv run --script`` builds from
the inline metadata above::

    uv run --script packages/astrogwb/scripts/generate_mock_population_fixture.py

``gwmock-pop`` is pinned *exactly* here even though ``astrogwb`` depends on it
with a floor (``>=0.11.4``): the graph YAML's own header warns that
``GraphSimulator`` draws RNG keys in topological order and breaks ties by
declaration order, so a patch release that reorders anything would resample the
population. The fixture has to be reproducible from this script alone.

Why a committed CSV rather than simulating in ``conftest.py``:

* The pinned graph lives in ``packages/astrogwb-paper/config/populations/``,
  and core "must never import ``astrogwb_paper`` or know the repository
  checkout layout" (AGENTS.md). Reaching for that path from a core conftest
  breaks the isolated ``--package astrogwb`` test run, which never installs the
  paper package. This script sits outside the installed package tree and is
  never imported by a test, so it may reference the YAML freely.
* Determinism: a ``gwmock-pop`` patch release could reshuffle the draw and flip
  a tolerance-based convergence assertion for reasons unrelated to this repo.

``seed=41`` matches ``config/banks/md-imrphenom-s41.toml``, the production
injection bank, and ``GraphSimulator`` draws from its construction-time RNG --
a property pinned by
``astrogwb-paper/tests/test_bank_generation.py::test_generation_draws_are_a_prefix_stable_stream``.
The fixture is therefore *literally the first N sources of the production
injection bank*, not merely a population drawn from the same graph.

Only regenerate this when intentionally updating the reference (for example, to
bump the pinned ``gwmock-pop``); astrogwb's own tests treat it as a fixed
external oracle, not a target to chase.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from gwmock_pop import GraphSimulator

#: Sources drawn. ~100 KB of CSV at ``%.17g``. The MCMC integration tests
#: subsample the first 256; the analytic cross-check uses all of them, where
#: the ~3% Monte-Carlo error sets the tolerance.
N_FIXTURE = 1024

#: The injection bank's seed; see the module docstring.
SEED = 41

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

DEFAULT_POPULATION_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "astrogwb-paper"
    / "config"
    / "populations"
    / "madau-dickinson.yaml"
)
OUT_PATH = (
    pathlib.Path(__file__).parent.parent
    / "tests"
    / "fixtures"
    / "mock_bns_population.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw the committed mock BNS population fixture from a gwmock-pop "
            "graph config."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "population",
        nargs="?",
        type=pathlib.Path,
        default=DEFAULT_POPULATION_PATH,
        help="path to the population graph YAML",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=OUT_PATH,
        help="path to write the fixture CSV to",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=N_FIXTURE,
        help="number of sources to draw",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="GraphSimulator seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.population.is_file():
        raise SystemExit(f"population graph not found: {args.population}")

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
        f"population={args.population.stem} seed={args.seed} "
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
