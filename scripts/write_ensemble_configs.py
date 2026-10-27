"""Write the simulation inputs of the paper's ensembles from the paper config.

``astrogwb simulate`` reads its stage settings from TOML. This writes them once
from the shared config layers, to be committed:

- ``config/networks.toml`` -- every paper network, ``--background-metadata``;
- ``config/ensembles/<name>/population.toml`` and ``waveform.toml`` -- each
  ensemble's ``--population-metadata`` and ``--waveform-metadata``.

Rerun it after changing a detector, the fiducials, the population or the
waveform; ``tests/paper/test_ensemble.py`` fails while the committed files are
stale. Existing ensembles are then regenerated, not migrated.

    uv run python scripts/write_ensemble_configs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tomli_w

from astrogwb.paper.ensemble import ensemble_config_tables

ROOT_DIR = Path(__file__).resolve().parents[1]
HEADER = "# Written by scripts/write_ensemble_configs.py; do not edit.\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT_DIR,
        help="repository root (default: %(default)s)",
    )
    args = parser.parse_args()
    for relative, table in ensemble_config_tables(args.root).items():
        path = args.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER + tomli_w.dumps(table))
        print(path)


if __name__ == "__main__":
    main()
