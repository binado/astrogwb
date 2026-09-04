"""Generate one reusable waveform catalog from its config layers.

One rule, one file: this merges ``config/catalogs/base/*.toml`` with
``config/catalogs/defs/<catalog>.toml``, simulates that catalog's population
component(s) in-process, generates frequency-domain waveforms with the Ripple
backend, reduces them to polarization power, and writes
``outputs/catalogs/<catalog>.h5``.

The population never lands on disk. It used to be a ``temp()`` workflow node
with exactly one consumer, and keeping it would have forced the intermediate to
be keyed by *catalog* rather than population -- several catalogs share one
graph at different seeds and sizes. Simulating in-process makes the population
config one-to-one and shared, and removes a node from the DAG.

A catalog with one component draws straight through ``GraphSimulator``; two or
more go through ``MixtureSimulator``. That dispatch is load-bearing, not an
optimization -- see :func:`draw_population`.

The catalog also records how it was made: seed, sample count, and the redshift
proposal *mixture* its samples follow (see
:mod:`astrogwb.paper.config.catalogs`). The graph YAMLs are interpreted as
densities here, once, and never again -- the analysis reads the recorded
descriptor.

Usage::

    uv run --extra paper python scripts/generate_catalog.py \\
        --config config/catalogs/base/waveform.toml \\
        --config config/catalogs/defs/md-imrphenom-s41-n32768.toml \\
        --output outputs/catalogs/md-imrphenom-s41-n32768.h5
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from gwmock_signal.waveform import RippleBackend

from astrogwb.catalog import (
    Catalog,
    FrequencyDomainWaveformMetadata,
    PopulationMetadata,
    simulate_population,
    simulate_population_mixture,
)
from astrogwb.catalog.io import save_catalog
from astrogwb.paper.config.catalogs import (
    CatalogDefinition,
    CatalogProvenance,
    extract_mixture_proposal,
    load_catalog_layers,
)
from astrogwb.paper.utils import load_mapping
from astrogwb.waveform import polarization_power

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a BNS population from its graph config(s), generate "
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


def _resolve_segment_duration(frequency_resolution: float) -> float:
    """Map a target frequency resolution to a segment duration in seconds.

    df = 1 / segment_duration -- two views of the same knob. We work with the
    polarizations purely in the frequency domain (never an inverse FFT), so a
    coarse df is safe: the long, fine-df segment the backend would otherwise
    pick exists only to avoid time-domain wraparound of the inspiral, which
    cannot affect a frequency-domain-only catalog.
    """
    if frequency_resolution <= 0:
        raise ValueError("waveform.frequency_resolution must be > 0")
    return 1.0 / frequency_resolution


def _effective_resolution(segment_duration: float, sampling_frequency: float) -> float:
    """Achieved df after the backend rounds the segment up to a power-of-two seconds."""
    rounded_seconds = float(2.0 ** math.ceil(math.log2(segment_duration)))
    return sampling_frequency / round(rounded_seconds * sampling_frequency)


def _truncate(
    frequencies: np.ndarray,
    plus: np.ndarray,
    cross: np.ndarray,
    maximum_frequency: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restrict the frequency axis (and matching polarization columns) to f <= f_max.

    ``plus`` and ``cross`` are in the backend's ``(n_events, n_freq)`` orientation.
    Truncation happens here, before the polarization-power reduction.
    """
    if maximum_frequency is None:
        return frequencies, plus, cross
    mask = frequencies <= maximum_frequency
    return frequencies[mask], plus[:, mask], cross[:, mask]


def _generate_polarization_power(
    samples: dict[str, np.ndarray],
    *,
    approximant: str,
    sampling_frequency: float,
    minimum_frequency: float,
    backend: RippleBackend,
    maximum_frequency: float | None,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate polarization power chunk by chunk, reusing one fixed-grid backend.

    The backend already carries a fixed ``segment_duration``, so every chunk lands on
    the same frequency axis. Each chunk is truncated to ``maximum_frequency`` and then
    reduced to power immediately, so the running accumulator holds float64 power
    instead of two complex polarization arrays -- a 4x cut in peak memory -- and the
    chunks concatenate directly into a C-contiguous ``(n_freq, n_events)`` array with
    no full-size transpose ever materialized. Returns ``frequencies`` plus that power
    array.
    """
    n_events = samples["detector_frame_mass_1"].shape[0]
    step = chunk_size if 0 < chunk_size < n_events else n_events
    frequencies: np.ndarray | None = None
    power_chunks: list[np.ndarray] = []
    for start in range(0, n_events, step):
        stop = min(start + step, n_events)
        chunk_samples = {
            name: np.asarray(values)[start:stop] for name, values in samples.items()
        }
        polarizations = backend.generate_fd_polarizations_batch(
            approximant,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            parameters=chunk_samples,
        )
        chunk_frequencies, chunk_plus, chunk_cross = _truncate(
            np.asarray(polarizations.frequencies),
            np.asarray(polarizations.plus),
            np.asarray(polarizations.cross),
            maximum_frequency,
        )
        if frequencies is None:
            frequencies = chunk_frequencies
        chunk_power = polarization_power(chunk_plus, chunk_cross)  # (F, n_chunk)
        power_chunks.append(chunk_power)
        logger.info("Generated chunk %d:%d of %d events", start, stop, n_events)

    assert frequencies is not None  # n_events > 0 guaranteed by CatalogDefinition
    return frequencies, np.concatenate(power_chunks, axis=1)


def resolve_population_paths(
    definition: CatalogDefinition, config_path: Path
) -> tuple[Path, ...]:
    """Locate each component's population graph, in component order.

    Prefers the tree the catalog def itself sits in
    (``<root>/config/catalogs/defs/*``), so a config copied out for a smoke run
    still finds its graphs; falls back to the checked-out workspace.
    """
    sibling_root = config_path.parent.parent.parent
    resolved: list[Path] = []
    for component in definition.components:
        candidate = component.population_path(sibling_root)
        resolved.append(
            candidate if candidate.is_file() else component.population_path()
        )
    return tuple(resolved)


def catalog_provenance(
    definition: CatalogDefinition, population_paths: Sequence[Path]
) -> CatalogProvenance:
    """Derive what this catalog will record about its own generation.

    ``seed`` is ``draw_seed``: the mixture-assignment seed for a mixture and
    the lone component's own seed otherwise, matching what
    :func:`draw_population` actually seeds the draw with.
    """
    return CatalogProvenance(
        name=definition.name,
        seed=definition.draw_seed,
        num_samples=definition.num_samples,
        redshift_proposal=extract_mixture_proposal(
            [
                (load_mapping(path), component.weight)
                for path, component in zip(
                    population_paths, definition.components, strict=True
                )
            ]
        ),
    )


def draw_population(
    definition: CatalogDefinition,
    population_paths: Sequence[Path],
    *,
    metadata: PopulationMetadata,
) -> dict[str, np.ndarray]:
    """Simulate this catalog's source parameters, dispatching on component count.

    A lone component goes straight through ``GraphSimulator`` rather than
    through a one-entry ``MixtureSimulator``. That is not an optimization: a
    mixture splits its seed to draw component assignments, so wrapping a single
    graph would change its RNG stream. Keeping the direct path is what makes
    single-component catalogs prefix-stable across sizes -- the three
    ``md-imrphenom-s42-n*`` catalogs are nested draws, which is what makes
    variable-catalog-size a clean series rather than three unrelated runs.
    """
    if not definition.is_mixture:
        return simulate_population(population_paths[0], metadata=metadata)
    return simulate_population_mixture(
        [
            (path, component.seed, component.weight)
            for path, component in zip(
                population_paths, definition.components, strict=True
            )
        ],
        metadata=metadata,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args(argv)
    config_paths = [path.expanduser().resolve() for path in args.config]
    config_path = config_paths[-1]
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"refusing to replace existing catalog: {output_path}. "
            "Pass --force only for an intentional replacement."
        )

    definition = load_catalog_layers(config_paths)
    population_paths = resolve_population_paths(definition, config_path)
    provenance = catalog_provenance(definition, population_paths)
    population_metadata = provenance.to_population_metadata()
    logger.info(
        "Catalog %s: seed=%d num_samples=%d components=[%s]",
        definition.name,
        provenance.seed,
        definition.num_samples,
        ", ".join(
            f"{component.population}@s{component.seed}*{proposal.weight:.4g}"
            for component, proposal in zip(
                definition.components,
                provenance.redshift_proposal.components,
                strict=True,
            )
        ),
    )

    population = draw_population(
        definition, population_paths, metadata=population_metadata
    )

    # Source -> detector frame: redshift the component masses. gwmock provides only
    # the inverse conversion, so the (1 + z) scaling is applied inline here.
    one_plus_z = 1.0 + population["redshift"]
    # Keep the full population (redshift, source-frame masses, ...) so downstream
    # importance sampling retains them; the Ripple batch path reads only the
    # canonical keys it needs and ignores the extras.
    samples = {
        **population,
        "detector_frame_mass_1": population["source_frame_mass_1"] * one_plus_z,
        "detector_frame_mass_2": population["source_frame_mass_2"] * one_plus_z,
    }

    waveform = definition.waveform
    segment_duration = _resolve_segment_duration(waveform.frequency_resolution)
    effective_df = _effective_resolution(segment_duration, waveform.sampling_frequency)
    # A fixed segment_duration pins df at generation time so we evaluate the analytic
    # FD waveform directly on the coarse grid instead of generating ~1/df more bins and
    # discarding them. Safe here because the catalog is consumed in the frequency domain
    # only; this backend must NOT be used for time-domain (inverse-FFT) generation.
    backend = RippleBackend(
        f_ref=waveform.reference_frequency, segment_duration=segment_duration
    )
    logger.info(
        "Generating %s waveforms for %d events (f_min=%.1f Hz, f_ref=%.1f Hz, "
        "f_s=%.1f Hz, segment=%.4g s, df=%.4g Hz)",
        waveform.approximant,
        definition.num_samples,
        waveform.minimum_frequency,
        waveform.reference_frequency,
        waveform.sampling_frequency,
        segment_duration,
        effective_df,
    )

    frequencies, power = _generate_polarization_power(
        samples,
        approximant=waveform.approximant,
        sampling_frequency=waveform.sampling_frequency,
        minimum_frequency=waveform.minimum_frequency,
        backend=backend,
        maximum_frequency=waveform.maximum_frequency,
        chunk_size=waveform.chunk_size,
    )
    logger.info("Truncated frequency axis to f <= %.1f Hz", waveform.maximum_frequency)

    actual_df = (
        float(frequencies[1] - frequencies[0]) if frequencies.size > 1 else effective_df
    )
    waveform_metadata = FrequencyDomainWaveformMetadata(
        frequencies=frequencies,
        approximant=waveform.approximant,
        minimum_frequency=waveform.minimum_frequency,
        maximum_frequency=waveform.maximum_frequency,
        reference_frequency=waveform.reference_frequency,
        sampling_frequency=waveform.sampling_frequency,
        df=actual_df,
    )
    catalog = Catalog(
        source_parameters={
            name: np.asarray(values) for name, values in samples.items()
        },
        polarization_power=power,
        waveform_metadata=waveform_metadata,
        population_metadata=population_metadata,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    save_catalog(output_path, catalog)

    logger.info(
        "Saved catalog %s: %d events, %d frequencies (%.2f-%.2f Hz), approximant=%s",
        definition.name,
        catalog.population_metadata.num_samples,
        catalog.waveform_metadata.frequencies.size,
        float(frequencies[0]),
        float(frequencies[-1]),
        waveform.approximant,
    )
    logger.info("Output written to %s", output_path)


if __name__ == "__main__":
    main()
