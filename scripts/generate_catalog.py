"""Generate one reusable waveform catalog from its merged config blocks.

One rule, one file: this takes the already-merged ``[population]``,
``[fiducials]`` and ``[waveform]`` blocks, draws that catalog's population from
the registered NumPyro model it names, generates frequency-domain waveforms,
reduces them to polarization power, and writes
``outputs/catalogs/<catalog>.h5``.

The layers are JSON, so the merge itself belongs to the caller. The
``Snakefile`` gives it its own rule, ``merge_catalog_config``, which folds the
files it declares as ``input:`` into one ``temp()`` JSON file; this script's
caller then reads a key per block out of that. That is a plain recursive merge
-- the catalog layers carry no ``[priors]`` block, so the shallow-merge rule
:func:`~astrogwb.paper.config.runs._merge_run_overlay` exists for never applies
here, and ``jq``'s ``*`` is :func:`~astrogwb.paper.utils.deep_merge` exactly.
Validation still happens in one place: the blocks are handed to
:class:`~astrogwb.paper.config.catalogs.CatalogDefinition` whole.

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

    layers="config/waveform.json config/population.json config/fiducials.json \\
        config/catalogs/md-imrphenom-s41-n32768.json"
    merged=outputs/catalogs/md-imrphenom-s41-n32768.merged.json

    jq -s 'reduce .[] as $layer ({}; . * $layer)' $layers > "$merged"

    uv run --extra paper python scripts/generate_catalog.py \\
        --name md-imrphenom-s41-n32768 \\
        --population "$(jq -c .population "$merged")" \\
        --fiducials "$(jq -c .fiducials "$merged")" \\
        --waveform "$(jq -c .waveform "$merged")" \\
        --seed 41 --num-samples 32768 \\
        --output outputs/catalogs/md-imrphenom-s41-n32768.h5
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.paper.config.catalogs import CatalogDefinition, check_population_model
from astrogwb.utils.sampling import sample_sources

# x64 must be on before the population draw. `build_catalog` samples before it
# builds the Ripple-backed generator, and importing ripplegw -- which turns this
# on globally -- happens only inside that generator, so relying on it would draw
# and persist every source column in float32.
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


def _json_object(raw: str) -> dict[str, Any]:
    """Parse one config block off argv, rejecting anything but an object."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"not valid JSON: {error}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {type(value)}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a BNS population from its registered NumPyro model, generate "
            "frequency-domain waveforms, and persist the polarization power as "
            "an astrogwb_catalog HDF5 file."
        )
    )
    parser.add_argument(
        "--name",
        required=True,
        help=(
            "The catalog's name: the stem of config/catalogs/<name>.json, and "
            "of the .h5 it produces."
        ),
    )
    parser.add_argument(
        "--population",
        required=True,
        type=_json_object,
        metavar="JSON",
        help=(
            "The merged [population] block: model_name and model_kwargs. The "
            "seed and the density sites are not configuration and are supplied "
            "from --seed and the registry."
        ),
    )
    parser.add_argument(
        "--fiducials",
        required=True,
        type=_json_object,
        metavar="JSON",
        help="The merged [fiducials] block: the hyperparameters to draw at.",
    )
    parser.add_argument(
        "--waveform",
        required=True,
        type=_json_object,
        metavar="JSON",
        help="The merged [waveform] block: the generator's settings.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="The population draw's seed.",
    )
    parser.add_argument(
        "--num-samples",
        required=True,
        type=int,
        help="How many sources to draw.",
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
        population.model_name,
        label=f"catalog {definition.name!r} population.model_name",
        kwargs=population.model_kwargs,
    )
    source_model = population.build().source_model

    logger.info(
        "Catalog %s: population=%s seed=%d num_samples=%d kwargs=%s",
        definition.name,
        population.model_name,
        definition.seed,
        definition.num_samples,
        population.model_kwargs,
    )
    samples = sample_sources(
        source_model,
        jax.random.PRNGKey(definition.seed),
        definition.fiducials,
        num_samples=definition.num_samples,
    )

    generator = definition.waveform.build()
    logger.info(
        "Generating %s waveforms for %d events (f_min=%.1f Hz, f_ref=%.1f Hz, "
        "f_s=%.1f Hz)",
        generator.approximant,
        definition.num_samples,
        generator.minimum_frequency,
        generator.reference_frequency,
        generator.sampling_frequency,
    )
    segment_duration = getattr(generator, "segment_duration", None)
    n_samples = getattr(generator, "n_samples", None)
    if segment_duration is not None and n_samples is not None:
        logger.info("Grid: segment=%.4g s, n=%d", segment_duration, n_samples)
    logger.info("Truncated frequency axis to f <= %.1f Hz", generator.maximum_frequency)

    # Values are checked once, here, on the concrete catalog: generation is
    # trace-safe and therefore trusts its inputs, so a population carrying a
    # degree of freedom this approximant cannot represent would otherwise be
    # silently dropped rather than reported.
    generator.check_sources(samples)

    catalog = PolarizationPowerCatalog.from_generator(
        samples,
        generator=generator,
        population=population,
        fiducials=definition.fiducials,
    )
    logger.info("Generated catalog with measured df=%.4g Hz", catalog.df)
    return catalog


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"refusing to replace existing catalog: {output_path}. "
            "Pass --force only for an intentional replacement."
        )

    definition = CatalogDefinition.model_validate(
        {
            "name": args.name,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "population": args.population,
            "fiducials": args.fiducials,
            "waveform": args.waveform,
        }
    )
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
