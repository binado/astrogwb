from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
from astrogwb.detector import load_sensitivity_map, resolve_detector
from astrogwb.resolved import optimal_snr
from pluscross import load_catalog

logger = logging.getLogger(__name__)

_DEFAULT_PROGRESS_LOG_EVERY = 1000


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a pluscross waveform catalog, generate fresh time-domain "
            "waveforms for every event using its waveform metadata, project "
            "them onto a detector network, and compute the per-detector optimal "
            "SNR with astrogwb.resolved.optimal_snr. The resulting "
            "(n_events, n_detectors) array is written to --output as dataset "
            "/snr with catalog_filename and detectors attributes."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help=(
            "Path to a pluscross catalog containing source parameters and "
            "waveform-generation metadata."
        ),
    )
    parser.add_argument(
        "--detectors",
        type=str,
        required=True,
        help=(
            "Comma-separated detector names, e.g. 'H1,L1,V1'. Column j of the "
            "output /snr dataset is the j-th name in this list."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .h5 for the SNR table.",
    )
    parser.add_argument(
        "--progress-log-every",
        type=_positive_int,
        default=_DEFAULT_PROGRESS_LOG_EVERY,
        metavar="EVENTS",
        help=(
            "Log progress every EVENTS completed events "
            f"(default: {_DEFAULT_PROGRESS_LOG_EVERY})."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "ripple", "lal"),
        default="auto",
        help=(
            "Waveform generation backend. 'auto' uses ripple when it supports "
            "the approximant, otherwise LALSimulation."
        ),
    )
    parser.add_argument(
        "--no-earth-rotation",
        action="store_true",
        help=(
            "Evaluate antenna patterns and delays once at the segment midpoint "
            "instead of at per-sample GPS times (faster, less accurate)."
        ),
    )
    return parser.parse_args()


def _progress_callback(log_every: int) -> Callable[[int, int], None]:
    def report(done: int, total: int) -> None:
        if done == total or done == 1 or done % log_every == 0:
            logger.info("Computed SNR for %d/%d events", done, total)

    return report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()
    catalog_path = args.catalog.resolve()

    detector_names = [
        name.strip() for name in args.detectors.split(",") if name.strip()
    ]
    if not detector_names:
        raise SystemExit("--detectors must name at least one detector.")
    logger.info("Resolving detectors %s", detector_names)
    detectors = [resolve_detector(name) for name in detector_names]
    sensitivities = load_sensitivity_map(detector_names)

    catalog = load_catalog(catalog_path)
    if not catalog.approximant:
        raise SystemExit("Catalog approximant must not be empty.")
    if catalog.sampling_frequency <= 0:
        raise SystemExit("Catalog sampling_frequency must be positive.")
    if catalog.minimum_frequency <= 0:
        raise SystemExit("Catalog minimum_frequency must be positive.")
    if catalog.maximum_frequency <= catalog.minimum_frequency:
        raise SystemExit(
            "Catalog maximum_frequency must be greater than minimum_frequency."
        )

    logger.info("Loaded %d events from %s", catalog.nsamples, catalog_path)
    logger.info(
        "Waveform model %s, sampling %.1f Hz, band [%.1f, %.1f] Hz",
        catalog.approximant,
        catalog.sampling_frequency,
        catalog.minimum_frequency,
        catalog.maximum_frequency,
    )

    snrs = optimal_snr(
        catalog.source_parameters,
        detectors,
        sensitivities,
        waveform_model=catalog.approximant,
        sampling_frequency=catalog.sampling_frequency,
        minimum_frequency=catalog.minimum_frequency,
        maximum_frequency=catalog.maximum_frequency,
        earth_rotation=not args.no_earth_rotation,
        backend=args.backend,
        progress_callback=_progress_callback(args.progress_log_every),
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as output:
        output.create_dataset("snr", data=snrs)
        output.attrs["catalog_filename"] = str(catalog_path)
        output.attrs["detectors"] = np.asarray(detector_names, dtype="S")
    logger.info("Wrote /snr with shape %s to %s", snrs.shape, output_path.resolve())


if __name__ == "__main__":
    main()
