"""Draw forward-model spectra from their metadata and save them under its key.

The draws are fully determined by a :class:`~astrogwb.metadata.SpectraMetadata`
-- waveform, population and seed, each hyperparameter's fixed value or prior,
draw count, observation time and plate depth -- handed in as JSON. This script
runs :class:`~astrogwb.catalog.SpectrumGenerator` on it and writes the result
atomically. It writes ``(draws, F)`` spectra only; no ``(F, N)`` catalog power
is ever materialized.

The output's stem must be the metadata's key, so a file can never be filed
under another record's address. Outside a shell,
``astrogwb.catalog.simulate(metadata, SpectrumGenerator(), cache_dir)`` is the
same generator behind a cache lookup, and needs no script.

A hyperparameter is a number to fix it, or a ``{"dist", "kwargs"}`` prior
(the format of the shared ``[priors]`` table) to draw it once per row.

Usage::

    uv run --extra io python scripts/simulate_spectra.py \\
        --metadata "$(cat spectra.json)" \\
        --output outputs/spectra/<key>.h5

where ``spectra.json`` is, for example::

    {
      "waveform": {"approximant": "TaylorF2", "sampling_frequency": 128.0,
                   "minimum_frequency": 20.0, "maximum_frequency": 48.0,
                   "reference_frequency": 20.0, "frequency_resolution": 4.0},
      "population": {"model_name": "bns_md_cosmological", "seed": 0,
                     "model_kwargs": {"minimum_redshift": 0.3,
                                      "maximum_redshift": 20, "n_grid": 64}},
      "hyperparameters": {"H0": 67.66, "Omega_m": 0.3096, "gamma": 1.42,
                          "kappa": 4.62, "z_peak": 1.84, "minimum_mass": 1.0,
                          "mass_width": 1.5,
                          "local_merger_rate": {"dist": "Normal",
                                                "kwargs": {"loc": 770.0, "scale": 7.7}}},
      "num_draws": 8,
      "observation_time": 1.0
    }
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from astrogwb.catalog import SpectrumGenerator, save_atomically
from astrogwb.metadata import SpectraMetadata

logger = logging.getLogger(__name__)


def _metadata(raw: str) -> SpectraMetadata:
    """Parse and validate the metadata off argv."""
    try:
        return SpectraMetadata.model_validate_json(raw)
    except ValidationError as error:
        raise argparse.ArgumentTypeError(f"invalid spectra metadata: {error}") from None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw forward-model spectral densities from their SpectraMetadata "
            "and save (draws, F) without materializing catalog power."
        )
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=_metadata,
        metavar="JSON",
        help="The SpectraMetadata, as JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .h5; its stem must be the metadata's key.",
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


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    metadata: SpectraMetadata = args.metadata
    output = args.output.expanduser().resolve()
    if output.stem != metadata.key():
        raise ValueError(
            f"output {output.name} is not named by the metadata's key {metadata.key()}"
        )
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
