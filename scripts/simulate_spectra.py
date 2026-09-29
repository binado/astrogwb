"""Draw forward-model spectra from config layers and save them under their key.

The draws are fully determined by a :class:`~astrogwb.metadata.SpectraMetadata`
-- waveform, population and seed, each hyperparameter's fixed value or prior,
draw count, observation time and plate depth -- declared as the ``[spectra]``
table of the ``--config`` layers, merged in process exactly as ``run_mcmc``
merges a run. This script runs :class:`~astrogwb.catalog.SpectrumGenerator` on
it and writes the result atomically to ``<output-dir>/<key>.h5``. It writes
``(draws, F)`` spectra only; no ``(F, N)`` catalog power is ever materialized.

The file is named by the metadata's key, so it can never be filed under another
record's address. Outside a shell,
``astrogwb.catalog.simulate(metadata, SpectrumGenerator(), cache_dir)`` is the
same generator behind a cache lookup, and needs no script.

A hyperparameter is a ``"${fiducials.X}"`` reference to fix it, or a
``"${priors.X}"`` one to draw it once per row; see
``config/simulations/spectrum/default.toml``.

Usage -- one ``--config`` per layer, in merge order::

    uv run --extra paper python scripts/simulate_spectra.py \\
        --config config/defaults.toml --config config/waveforms.toml \\
        --config config/populations.toml --config config/detectors.toml \\
        --config config/simulations/spectrum/default.toml

``knf <layers> --shallow 'priors.*' --interpolate`` prints the merged config,
whose ``[spectra]`` table is the record.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from astrogwb.catalog import SpectrumGenerator, save_atomically
from astrogwb.metadata import SpectraMetadata, artifact_path
from astrogwb.paper.config.runs import (
    SPECTRA_ROOT,
    add_config_arguments,
    load_merged_config,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw forward-model spectral densities from the [spectra] table of "
            "their config layers and save (draws, F) without materializing "
            "catalog power."
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPECTRA_ROOT,
        metavar="DIR",
        help=f"The spectra are written to <DIR>/<key>.h5 (default: {SPECTRA_ROOT}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Sources per waveform chunk; changes memory, not the draws.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output spectra file.",
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


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    metadata = spectra_metadata(load_merged_config(args))
    output = artifact_path(metadata, args.output_dir.expanduser()).resolve()
    if output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to replace existing spectra: {output}. "
            "Pass --force only for an intentional replacement."
        )

    catalog = SpectrumGenerator(batch_size=args.batch_size)(metadata)
    save_atomically(catalog, output)
    logger.info(
        "Saved spectra %s: %d draws, %d frequencies, to %s",
        metadata.key(),
        catalog.num_draws,
        catalog.frequencies.size,
        output,
    )


if __name__ == "__main__":
    main()
