"""Inspect the content-addressed catalog cache.

Catalog files are named by key -- ``outputs/catalogs/<key>.h5`` -- which is
unambiguous but unreadable. ``ls`` maps the keys back to what they hold: every
catalog a committed run asks for, what it draws, whether it has been built, and
which runs sample against it. ``--orphans`` lists built files no run asks for
any more (typically left behind by a config edit or a version bump), which are
safe to delete.

JAX-free: it resolves configs and reads nothing but directory listings.

Usage::

    uv run --extra paper python scripts/catalogs.py ls
    uv run --extra paper python scripts/catalogs.py ls --orphans
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from astrogwb.metadata import CatalogRequest
from astrogwb.paper.config.catalogs import RunCatalogs, resolve_run_catalogs
from astrogwb.paper.config.runs import CATALOGS_ROOT

#: Construction kwargs every catalog shares by default, left out of the
#: summary so what distinguishes one catalog from another stands out.
_SHARED_KWARGS = frozenset({"minimum_redshift", "maximum_redshift", "n_grid"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "ls", help="List every catalog the committed runs ask for."
    )
    listing.add_argument(
        "--cache-dir",
        type=Path,
        default=CATALOGS_ROOT,
        help=f"Where catalogs are built (default: {CATALOGS_ROOT}).",
    )
    listing.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root holding config/ (default: the working directory).",
    )
    listing.add_argument(
        "--orphans",
        action="store_true",
        help="List built catalogs no committed run asks for, instead.",
    )
    return parser.parse_args(argv)


def describe(request: CatalogRequest) -> str:
    """One line saying what a request draws."""
    population = request.metadata.population
    kwargs = ", ".join(
        f"{name}={value:g}"
        for name, value in sorted(population.model_kwargs.items())
        if name not in _SHARED_KWARGS
    )
    model = f"{population.model_name}({kwargs})" if kwargs else population.model_name
    return (
        f"{model} seed={population.seed} n={request.num_samples} "
        f"{request.metadata.waveform.approximant} v{request.version}"
    )


def format_listing(catalogs: RunCatalogs, cache_dir: Path) -> list[str]:
    """The ``ls`` table: one block per key, built ones marked."""
    lines: list[str] = []
    for key, request in sorted(
        catalogs.requests.items(), key=lambda item: describe(item[1])
    ):
        built = "built" if (cache_dir / f"{key}.h5").is_file() else "missing"
        lines.append(f"{key}  {built:<7}  {describe(request)}")
        lines.extend(f"    {user}" for user in catalogs.users(key))
    return lines


def orphans(catalogs: RunCatalogs, cache_dir: Path) -> list[Path]:
    """Built catalog files whose key no committed run resolves to."""
    if not cache_dir.is_dir():
        return []
    return sorted(
        path for path in cache_dir.glob("*.h5") if path.stem not in catalogs.requests
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    catalogs = resolve_run_catalogs(args.root)
    if args.orphans:
        for path in orphans(catalogs, args.cache_dir):
            print(path)
        return
    print("\n".join(format_listing(catalogs, args.cache_dir)))


if __name__ == "__main__":
    sys.exit(main())
