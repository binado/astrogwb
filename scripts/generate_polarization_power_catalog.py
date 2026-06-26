from __future__ import annotations

import argparse
import logging
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
    return parser.parse_args()


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

    backend = RippleBackend(f_ref=args.reference_frequency)
    logger.info(
        "Generating %s waveforms for %d events (f_min=%.1f Hz, f_ref=%.1f Hz, "
        "f_s=%.1f Hz)",
        args.approximant,
        detector_frame_mass_1.shape[0],
        args.minimum_frequency,
        args.reference_frequency,
        args.sampling_frequency,
    )
    catalog = generate_catalog_polarization_power(
        samples,
        approximant=args.approximant,
        sampling_frequency=args.sampling_frequency,
        minimum_frequency=args.minimum_frequency,
        backend=backend,
    )

    if args.maximum_frequency is not None:
        frequencies = np.asarray(catalog.frequencies)
        mask = frequencies <= args.maximum_frequency
        catalog = PolarizationPowerCatalog(
            frequencies=frequencies[mask],
            polarization_power=np.asarray(catalog.polarization_power)[mask, :],
            samples=catalog.samples,
        )
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
