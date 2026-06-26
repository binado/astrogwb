from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
from gwmock_pop.loaders.file_loader import read_population_catalogue
from gwmock_signal.waveform import RippleBackend

from astrogwb.waveform import (
    PolarizationPowerCatalog,
    generate_catalog_polarization_power,
    save_polarization_power_catalog,
)

logger = logging.getLogger(__name__)

#: Default frequency resolution in Hz (segment_duration = 1 / DEFAULT_FREQUENCY_RESOLUTION).
DEFAULT_FREQUENCY_RESOLUTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a gwmock-pop population file, convert source-frame masses to the "
            "detector frame, generate frequency-domain waveforms with the Ripple "
            "backend, and persist the polarization-power catalog."
        )
    )
    parser.add_argument(
        "--population",
        type=Path,
        required=True,
        help="Path to the .h5/.csv population produced by `gwmock-pop simulate`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/bns_polarization_power_catalog.npz"),
        help="Destination .npz for the polarization-power catalog.",
    )
    parser.add_argument(
        "--approximant",
        default="IMRPhenomXAS_NRTidalv3",
        help="Waveform approximant (aligned-spin + NRTidal tides).",
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        default=8192.0,
        help="Time-domain sampling frequency in Hz; Nyquist = sampling_frequency / 2.",
    )
    parser.add_argument(
        "--minimum-frequency",
        type=float,
        default=5.0,
        help="Lower bound of the generated frequency axis in Hz.",
    )
    parser.add_argument(
        "--reference-frequency",
        type=float,
        default=20.0,
        help="Reference frequency in Hz for the waveform phase/spin convention.",
    )
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        default=None,
        help=(
            "Optional upper bound in Hz. The backend always generates up to Nyquist; "
            "if set, the catalog frequency axis is truncated to f <= maximum_frequency."
        ),
    )
    # The frequency resolution df = 1 / segment_duration. These are two views of the
    # same knob, so they are mutually exclusive. We work with the polarizations purely
    # in the frequency domain (never an inverse FFT), so a coarse df is safe: the long,
    # fine-df segment the backend would otherwise pick exists only to avoid time-domain
    # wraparound of the inspiral, which cannot affect a frequency-domain-only catalog.
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument(
        "--frequency-resolution",
        type=float,
        default=None,
        help=(
            "Target frequency resolution df in Hz (= 1 / segment_duration). "
            f"Defaults to {DEFAULT_FREQUENCY_RESOLUTION} Hz. The backend rounds the "
            "implied segment up to a power-of-two seconds, so the achieved df is the "
            "nearest 1 / 2**k at or below the request. Mutually exclusive with "
            "--segment-duration."
        ),
    )
    grid.add_argument(
        "--segment-duration",
        type=float,
        default=None,
        help=(
            "Time-domain segment length in seconds (= 1 / df), rounded up to a "
            "power-of-two seconds. Mutually exclusive with --frequency-resolution."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "Generate the catalog in groups of this many events instead of one "
            "vmap over the whole population. Bounds the peak memory of the batched "
            "waveform evaluation, which scales as chunk_size x n_frequencies. "
            "Use 0 (default) or a value >= n_events for a single batch."
        ),
    )
    return parser.parse_args()


def _resolve_segment_duration(args: argparse.Namespace) -> float:
    """Map the mutually-exclusive resolution/segment flags to a segment duration (s)."""
    if args.segment_duration is not None:
        return args.segment_duration
    resolution = (
        args.frequency_resolution
        if args.frequency_resolution is not None
        else DEFAULT_FREQUENCY_RESOLUTION
    )
    if resolution <= 0:
        raise ValueError("--frequency-resolution must be > 0")
    return 1.0 / resolution


def _effective_resolution(segment_duration: float, sampling_frequency: float) -> float:
    """Achieved df after the backend rounds the segment up to a power-of-two seconds."""
    rounded_seconds = float(2.0 ** math.ceil(math.log2(segment_duration)))
    return sampling_frequency / round(rounded_seconds * sampling_frequency)


def _truncate(
    frequencies: np.ndarray,
    polarization_power: np.ndarray,
    maximum_frequency: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict the frequency axis (and matching power rows) to f <= maximum_frequency."""
    if maximum_frequency is None:
        return frequencies, polarization_power
    mask = frequencies <= maximum_frequency
    return frequencies[mask], polarization_power[mask, :]


def _generate_chunked(
    samples: dict[str, np.ndarray],
    *,
    approximant: str,
    sampling_frequency: float,
    minimum_frequency: float,
    backend: RippleBackend,
    maximum_frequency: float | None,
    chunk_size: int,
) -> PolarizationPowerCatalog:
    """Generate the catalog one event-chunk at a time, reusing one fixed-grid backend.

    The backend already carries a fixed ``segment_duration``, so every chunk lands on
    the same frequency axis and the chunks concatenate directly along the event axis.
    Each chunk is truncated to ``maximum_frequency`` before being accumulated, so the
    running ``polarization_power`` stays bounded too.
    """
    n_events = samples["detector_frame_mass_1"].shape[0]
    frequencies: np.ndarray | None = None
    power_chunks: list[np.ndarray] = []
    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        chunk_samples = {
            name: np.asarray(values)[start:stop] for name, values in samples.items()
        }
        chunk_catalog = generate_catalog_polarization_power(
            chunk_samples,
            approximant=approximant,
            sampling_frequency=sampling_frequency,
            minimum_frequency=minimum_frequency,
            backend=backend,
        )
        chunk_frequencies, chunk_power = _truncate(
            np.asarray(chunk_catalog.frequencies),
            np.asarray(chunk_catalog.polarization_power),
            maximum_frequency,
        )
        if frequencies is None:
            frequencies = chunk_frequencies
        power_chunks.append(chunk_power)
        logger.info("Generated chunk %d:%d of %d events", start, stop, n_events)

    assert frequencies is not None  # n_events > 0 guaranteed by the population loader
    return PolarizationPowerCatalog(
        frequencies=frequencies,
        polarization_power=np.concatenate(power_chunks, axis=1),
        samples=dict(samples),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()

    logger.info("Loading population from %s", args.population)
    population = read_population_catalogue(args.population)

    # Source -> detector frame: redshift the component masses. gwmock provides only
    # the inverse conversion, so the (1 + z) scaling is applied inline here.
    one_plus_z = 1.0 + population["redshift"]
    detector_frame_mass_1 = population["source_frame_mass_1"] * one_plus_z
    detector_frame_mass_2 = population["source_frame_mass_2"] * one_plus_z

    # Keep the full population (redshift, source-frame masses, ...) so downstream
    # importance sampling retains them; the Ripple batch path reads only the
    # canonical keys it needs and ignores the extras.
    samples = {
        **population,
        "detector_frame_mass_1": detector_frame_mass_1,
        "detector_frame_mass_2": detector_frame_mass_2,
    }

    n_events = detector_frame_mass_1.shape[0]
    segment_duration = _resolve_segment_duration(args)
    effective_df = _effective_resolution(segment_duration, args.sampling_frequency)
    # A fixed segment_duration pins df at generation time so we evaluate the analytic
    # FD waveform directly on the coarse grid instead of generating ~1/df more bins and
    # discarding them. Safe here because the catalog is consumed in the frequency domain
    # only; this backend must NOT be used for time-domain (inverse-FFT) generation.
    backend = RippleBackend(
        f_ref=args.reference_frequency, segment_duration=segment_duration
    )
    logger.info(
        "Generating %s waveforms for %d events (f_min=%.1f Hz, f_ref=%.1f Hz, "
        "f_s=%.1f Hz, segment=%.4g s, df=%.4g Hz)",
        args.approximant,
        n_events,
        args.minimum_frequency,
        args.reference_frequency,
        args.sampling_frequency,
        segment_duration,
        effective_df,
    )

    if 0 < args.chunk_size < n_events:
        catalog = _generate_chunked(
            samples,
            approximant=args.approximant,
            sampling_frequency=args.sampling_frequency,
            minimum_frequency=args.minimum_frequency,
            backend=backend,
            maximum_frequency=args.maximum_frequency,
            chunk_size=args.chunk_size,
        )
    else:
        catalog = generate_catalog_polarization_power(
            samples,
            approximant=args.approximant,
            sampling_frequency=args.sampling_frequency,
            minimum_frequency=args.minimum_frequency,
            backend=backend,
        )
        frequencies, polarization_power = _truncate(
            np.asarray(catalog.frequencies),
            np.asarray(catalog.polarization_power),
            args.maximum_frequency,
        )
        catalog = PolarizationPowerCatalog(
            frequencies=frequencies,
            polarization_power=polarization_power,
            samples=catalog.samples,
        )

    if args.maximum_frequency is not None:
        logger.info("Truncated frequency axis to f <= %.1f Hz", args.maximum_frequency)

    frequencies = np.asarray(catalog.frequencies)
    polarization_power = np.asarray(catalog.polarization_power)
    n_freq, n_events = polarization_power.shape

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_polarization_power_catalog(args.output, catalog)

    logger.info(
        "Saved catalog: %d events, %d frequencies (%.2f-%.2f Hz), approximant=%s",
        n_events,
        n_freq,
        float(frequencies[0]),
        float(frequencies[-1]),
        args.approximant,
    )
    logger.info("Output written to %s", args.output)


if __name__ == "__main__":
    main()
