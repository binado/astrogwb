"""Generate one reusable waveform catalog from its config layers.

One rule, one file: this merges ``config/waveform.json`` and
``config/catalogs/base/*.toml`` with ``config/catalogs/defs/<catalog>.toml``,
draws that catalog's population from the registered NumPyro model it names,
generates frequency-domain waveforms, reduces them to polarization power, and
writes ``outputs/catalogs/<catalog>.h5``.

The population declaration is one registered name, not a graph config, and it
is the *same* population the analysis evaluates the proposal density with.
That is what makes the output self-describing: the file records the
population's registry name, its construction kwargs, the hyperparameters it
was drawn at, and the density factors included in importance weighting, which
is everything needed to reconstruct the map from hyperparameters to source
density. Nothing downstream re-reads these configs, and no run config restates
any of it.

It also retired the arithmetic that used to sit in this script. Detector-frame
masses were computed here, by hand, from source-frame masses and redshift --
so nothing checked them on the way back in. They are now
``numpyro.deterministic`` sites of the source model, recomputed from the
stochastic values on every evaluation.

Usage::

    uv run --extra paper python scripts/generate_catalog.py \\
        --config config/waveform.json \\
        --config config/catalogs/base/population.toml \\
        --config config/catalogs/defs/md-imrphenom-s41-n32768.toml \\
        --output outputs/catalogs/md-imrphenom-s41-n32768.h5
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import jax

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.paper.config.catalogs import (
    CatalogDefinition,
    check_population_model,
    load_catalog_layers,
)
from astrogwb.populations import build_population
from astrogwb.utils.sampling import sample_sources

# x64 must be on before the population draw. `build_catalog` samples before it
# builds the Ripple-backed generator, and importing ripplegw -- which turns this
# on globally -- happens only inside that generator, so relying on it would draw
# and persist every source column in float32.
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a BNS population from its registered NumPyro model, generate "
            "frequency-domain waveforms, and persist the polarization power as "
            "an astrogwb_catalog HDF5 file."
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


def build_catalog(definition: CatalogDefinition) -> PolarizationPowerCatalog:
    """Draw the population, generate its power, and record what produced it."""
    population = definition.population
    check_population_model(
        population.model,
        label=f"catalog {definition.name!r} population.model",
        kwargs=population.kwargs,
    )
    source_model = build_population(population.model, **population.kwargs).source_model

    logger.info(
        "Catalog %s: population=%s seed=%d num_samples=%d kwargs=%s",
        definition.name,
        population.model,
        definition.seed,
        definition.num_samples,
        population.kwargs,
    )
    samples = sample_sources(
        source_model,
        jax.random.PRNGKey(definition.seed),
        population.params,
        num_samples=definition.num_samples,
    )

    generator = definition.waveform.build()
    logger.info(
        "Generating %s waveforms for %d events (f_min=%.1f Hz, f_ref=%.1f Hz, "
        "f_s=%.1f Hz)",
        generator.metadata.approximant,
        definition.num_samples,
        generator.metadata.minimum_frequency,
        generator.metadata.reference_frequency,
        generator.metadata.sampling_frequency,
    )
    segment_duration = getattr(generator, "segment_duration", None)
    n_samples = getattr(generator, "n_samples", None)
    if segment_duration is not None and n_samples is not None:
        logger.info("Grid: segment=%.4g s, n=%d", segment_duration, n_samples)
    logger.info(
        "Truncated frequency axis to f <= %.1f Hz", generator.metadata.maximum_frequency
    )

    # Values are checked once, here, on the concrete catalog: generation is
    # trace-safe and therefore trusts its inputs, so a population carrying a
    # degree of freedom this approximant cannot represent would otherwise be
    # silently dropped rather than reported.
    generator.check_sources(samples)

    catalog = PolarizationPowerCatalog.from_generator(
        samples,
        generator=generator,
        population=definition.population_record(),
        fiducials=population.params,
    )
    logger.info("Generated catalog with measured df=%.4g Hz", catalog.df)
    return catalog


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
        catalog.num_samples,
        catalog.frequencies.size,
        catalog.frequencies[0].item(),
        catalog.frequencies[-1].item(),
        definition.waveform.approximant,
    )
    logger.info("Output written to %s", output_path)


if __name__ == "__main__":
    main()
