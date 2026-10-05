"""Draw forward-model spectra from config layers and save them under their key.

The draws are determined by a :class:`~astrogwb.simulators.spectra.SpectraMetadata`
-- waveform, population, each hyperparameter's fixed value or prior,
observation time, count mode and source count or padding -- declared as the
``[spectra]`` table of the ``--config`` layers, merged in process exactly as
``run_mcmc`` merges a run, and by the seeds the sibling ``[draws]`` table names
(``seed`` split into ``num_draws`` children). This script calls the cached
:func:`~astrogwb.simulators.spectra.spectra` node on them, which writes
``<output-dir>/spectra-<key>-<digest>.h5``. It writes ``(draws, F)`` spectra
only; no ``(F, N)`` catalog power is ever materialized. A second invocation
with the same layers is a cache hit.

A hyperparameter is a ``"${fiducials.X}"`` reference to fix it, or a
``"${priors.X}"`` one to draw it once per row; see
``config/simulations/spectrum/poisson.toml`` (Poisson) or ``fixed.toml``
(exactly ``num_events`` sources in each of ``num_draws`` realizations).

Usage -- one ``--config`` per layer, in merge order::

    uv run --extra paper python scripts/simulate_spectra.py \\
        --config config/defaults.toml --config config/waveforms.toml \\
        --config config/populations.toml --config config/detectors.toml \\
        --config config/simulations/spectrum/poisson.toml

``knf <layers> --shallow 'priors.*' --interpolate`` prints the merged config,
whose ``[spectra]`` table is the record and ``[draws]`` the seeds.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from astrogwb.paper.cache import default_cache_dir
from astrogwb.paper.config.runs import (
    add_config_arguments,
    load_merged_config,
)
from astrogwb.simulators.core import split_seed
from astrogwb.simulators.spectra import (
    SpectralDensityCatalog,
    SpectraMetadata,
    spectra,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw Poisson or fixed-count forward-model spectral densities from "
            "the [spectra] table of "
            "their config layers and save (draws, F) without materializing "
            "catalog power. Fixed mode requires num_events sources per realization; "
            "[draws].num_draws sets the realization count."
        )
    )
    add_config_arguments(
        parser,
        help=(
            "One config layer file, in merge order; repeat once per layer (the "
            "four shared config/*.toml layers, then a "
            "config/simulations/spectrum/<name>.toml). Its [spectra] table is "
            "the SpectraMetadata."
        ),
    )
    cache_dir = default_cache_dir() / "spectra"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=cache_dir,
        metavar="DIR",
        help=f"The cache directory (default: {cache_dir}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Sources per waveform chunk; changes memory, not the draws.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Draw again and replace an existing cached spectra file.",
    )
    return parser.parse_args(argv)


def spectra_metadata(config: dict[str, Any]) -> SpectraMetadata:
    """Validate the merged config's ``[spectra]`` table as a ``SpectraMetadata``.

    The whole merge is not the record: it also carries ``[analysis]``,
    ``[fiducials]``, ``[priors]`` and the rest, which the model forbids.
    """
    if "spectra" not in config:
        raise ValueError(
            "the merged config has no [spectra] table; add a "
            "config/simulations/spectrum/<name>.toml layer"
        )
    try:
        return SpectraMetadata.model_validate(config["spectra"])
    except ValidationError as error:
        raise ValueError(f"invalid [spectra] table: {error}") from None


def draw_seeds(config: dict[str, Any]) -> NDArray[np.uint64]:
    """The seeds the merged config's ``[draws]`` table names."""
    draws = config.get("draws")
    if not isinstance(draws, dict) or set(draws) != {"seed", "num_draws"}:
        raise ValueError(
            "the merged config needs a [draws] table with exactly seed and "
            "num_draws; add it to the config/simulations/spectrum/<name>.toml layer"
        )
    if draws["num_draws"] <= 0:
        raise ValueError("[draws].num_draws must be positive")
    return split_seed(draws["seed"], draws["num_draws"])


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    config = load_merged_config(args)
    metadata = spectra_metadata(config)
    inputs = {"seeds": draw_seeds(config)}
    cache_dir = args.output_dir.expanduser()
    output = spectra.path(inputs, metadata, cache_dir).resolve()
    if args.force:
        output.unlink(missing_ok=True)
    hit = output.exists()

    outputs = spectra(inputs, metadata, cache_dir=cache_dir, chunk_size=args.chunk_size)
    catalog = SpectralDensityCatalog.from_arrays(outputs, metadata)
    logger.info(
        "%s spectra %s: count=%s num_events=%s, %d draws, %d frequencies, at %s",
        "Cache hit for" if hit else "Saved",
        metadata.key(),
        metadata.count,
        metadata.num_events,
        catalog.num_draws,
        catalog.frequencies.size,
        output,
    )


if __name__ == "__main__":
    main()
