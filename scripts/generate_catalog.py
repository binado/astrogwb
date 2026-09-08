"""Generate one reusable waveform catalog from its config layers.

One rule, one file: this merges ``config/catalogs/base/*.toml`` with
``config/catalogs/defs/<catalog>.toml``, draws that catalog's population from
the registered NumPyro model it names, generates frequency-domain waveforms
with the Ripple backend, reduces them to polarization power, and writes
``outputs/catalogs/<catalog>.h5``.

The population declaration is a callable population, not a graph config, and it is
the *same* declaration the analysis evaluates the proposal density with. That
is what makes the output self-describing: the file records the model's registry
name, its construction settings, the hyperparameters it was drawn at, and the
density factors included in importance weighting, which is everything needed
to reconstruct the map from hyperparameters to source density. Nothing
downstream re-reads these configs, and no run config restates any of it.

It also retired the arithmetic that used to sit in this script. Detector-frame
masses were computed here, by hand, from source-frame masses and redshift --
so nothing checked them on the way back in. They are now
``numpyro.deterministic`` sites of the population, recomputed and compared
against the stored columns every time the catalog is loaded.

Usage::

    uv run --extra paper python scripts/generate_catalog.py \\
        --config config/catalogs/base/population.toml \\
        --config config/catalogs/base/waveform.toml \\
        --config config/catalogs/defs/md-imrphenom-s41-n32768.toml \\
        --output outputs/catalogs/md-imrphenom-s41-n32768.h5
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import jax

from astrogwb.catalog import Catalog, PopulationMetadata
from astrogwb.paper.config.catalogs import (
    CatalogDefinition,
    check_population_model,
    load_catalog_layers,
)
from astrogwb.populations import population_model
from astrogwb.waveform import RippleGenerator

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a BNS population from its registered NumPyro model, generate "
            "frequency-domain waveforms with the Ripple backend, and persist "
            "the polarization power as an astrogwb_catalog HDF5 file."
        )
    )
    parser.add_argument(
        "--config",
        dest="config",
        action="append",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "One catalog config layer, in merge order; repeat the flag. The "
            "last layer is config/catalogs/defs/<catalog>.toml and its stem "
            "names the catalog."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .h5 for the catalog.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output catalog.",
    )
    return parser.parse_args(argv)


def build_catalog(definition: CatalogDefinition) -> Catalog:
    """Draw the population, generate its power, and record what produced it."""
    population = definition.population
    check_population_model(
        population.model, label=f"catalog {definition.name!r} population.model"
    )
    model = population_model(population.model)(**population.kwargs)

    logger.info(
        "Catalog %s: model=%s seed=%d num_samples=%d kwargs=%s",
        definition.name,
        population.model,
        definition.seed,
        definition.num_samples,
        population.kwargs,
    )
    samples = model.sample(
        jax.random.PRNGKey(definition.seed),
        population.params,
        num_samples=definition.num_samples,
    )

    waveform = definition.waveform
    generator = RippleGenerator(
        approximant=waveform.approximant,
        sampling_frequency=waveform.sampling_frequency,
        minimum_frequency=waveform.minimum_frequency,
        maximum_frequency=waveform.maximum_frequency,
        reference_frequency=waveform.reference_frequency,
        frequency_resolution=waveform.frequency_resolution,
        chunk_size=waveform.chunk_size,
    )
    logger.info(
        "Generating %s waveforms for %d events (f_min=%.1f Hz, f_ref=%.1f Hz, "
        "f_s=%.1f Hz, segment=%.4g s, df=%.4g Hz)",
        waveform.approximant,
        definition.num_samples,
        waveform.minimum_frequency,
        waveform.reference_frequency,
        waveform.sampling_frequency,
        1.0 / generator.frequency_resolution,
        generator.df,
    )
    logger.info("Truncated frequency axis to f <= %.1f Hz", waveform.maximum_frequency)

    return Catalog.from_generator(
        samples,
        generator=generator,
        population_metadata=PopulationMetadata(
            name=population.model,
            seed=definition.seed,
            num_samples=definition.num_samples,
            source_type="bns",
        ),
        model_name=population.model,
        model_kwargs=population.kwargs,
        population_params=population.params,
        density_sites=model.density_sites,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    config_paths = [path.expanduser().resolve() for path in args.config]
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"refusing to replace existing catalog: {output_path}. "
            "Pass --force only for an intentional replacement."
        )

    definition = load_catalog_layers(config_paths)
    catalog = build_catalog(definition)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    catalog.save(output_path)

    logger.info(
        "Saved catalog %s: %d events, %d frequencies (%.2f-%.2f Hz), approximant=%s",
        definition.name,
        catalog.population_metadata.num_samples,
        catalog.waveform_metadata.frequencies.size,
        float(catalog.waveform_metadata.frequencies[0]),
        float(catalog.waveform_metadata.frequencies[-1]),
        definition.waveform.approximant,
    )
    logger.info("Output written to %s", output_path)


if __name__ == "__main__":
    main()
