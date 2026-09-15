"""Generate one reusable waveform catalog from its config layers.

One rule, one file: this merges ``config/catalogs/base/*.toml`` with
``config/catalogs/defs/<catalog>.toml``, draws that catalog's population from
the registered NumPyro model it names, generates frequency-domain waveforms
with the Ripple backend, reduces them to polarization power, and writes
``outputs/catalogs/<catalog>.h5``.

The population declaration is a source model composed with a merger-rate
model, not a graph config, and it is the *same* pair the analysis evaluates
the proposal density with. That is what makes the output self-describing: the
file records both models' registry names, their construction settings, the
hyperparameters they were drawn at, and the density factors included in
importance weighting, which is everything needed to reconstruct the map from
hyperparameters to source density. Nothing downstream re-reads these configs,
and no run config restates any of it.

It also retired the arithmetic that used to sit in this script. Detector-frame
masses were computed here, by hand, from source-frame masses and redshift --
so nothing checked them on the way back in. They are now
``numpyro.deterministic`` sites of the source model, recomputed from the
stochastic values on every evaluation.

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

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.paper.config.catalogs import (
    CatalogDefinition,
    check_rate_model,
    check_source_model,
    load_catalog_layers,
)
from astrogwb.populations import DEFAULT_DENSITY_SITES, build_source_model
from astrogwb.utils.sampling import sample_sources
from astrogwb.waveform import RippleGenerator

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


def build_catalog(definition: CatalogDefinition) -> PolarizationPowerCatalog:
    """Draw the population, generate its power, and record what produced it."""
    population = definition.population
    check_source_model(
        population.source_model,
        label=f"catalog {definition.name!r} population.source_model",
    )
    check_rate_model(
        population.rate_model,
        label=f"catalog {definition.name!r} population.rate_model",
    )
    source_model = build_source_model(
        population.source_model,
        settings=population.kwargs,
        source_kwargs=population.source_kwargs,
    )

    logger.info(
        "Catalog %s: source_model=%s rate_model=%s seed=%d num_samples=%d kwargs=%s "
        "source_kwargs=%s",
        definition.name,
        population.source_model,
        population.rate_model,
        definition.seed,
        definition.num_samples,
        population.kwargs,
        population.source_kwargs,
    )
    samples = sample_sources(
        source_model,
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
    )
    logger.info(
        "Generating %s waveforms for %d events (f_min=%.1f Hz, f_ref=%.1f Hz, "
        "f_s=%.1f Hz, segment=%.4g s, n=%d)",
        waveform.approximant,
        definition.num_samples,
        waveform.minimum_frequency,
        waveform.reference_frequency,
        waveform.sampling_frequency,
        generator.segment_duration,
        generator.n_samples,
    )
    logger.info("Truncated frequency axis to f <= %.1f Hz", waveform.maximum_frequency)

    # Values are checked once, here, on the concrete catalog: generation is
    # trace-safe and therefore trusts its inputs, so a population carrying a
    # degree of freedom this approximant cannot represent would otherwise be
    # silently dropped rather than reported.
    generator.check_sources(samples)

    catalog = PolarizationPowerCatalog.from_generator(
        samples,
        generator=generator,
        source_model_name=population.source_model,
        rate_model_name=population.rate_model,
        model_kwargs={**population.kwargs, **population.source_kwargs},
        fiducials=population.params,
        density_sites=DEFAULT_DENSITY_SITES,
        seed=definition.seed,
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
