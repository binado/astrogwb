from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astrogwb.detector import load_sensitivity_map, resolve_detector
from astrogwb.resolved.snr import optimal_snr
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

#: Source-parameter keys the SNR pipeline itself requires (mirrors
#: astrogwb.resolved.snr._REQUIRED_PARAMETER_KEYS); the check lives here so the
#: CLI can report a clearer error naming the input catalog.
REQUIRED_SOURCE_PARAMETERS = (
    "tc",
    "ra",
    "dec",
    "psi",
    "detector_frame_mass_1",
    "detector_frame_mass_2",
    "luminosity_distance",
)

#: Logging cadence for the progress callback, in events.
PROGRESS_LOG_EVERY = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the /source_parameters group of a catalog .h5 file, generate "
            "fresh time-domain waveforms for every event, project them onto a "
            "detector network, and compute the per-detector optimal SNR with "
            "astrogwb.resolved.optimal_snr. The resulting "
            "(n_events, n_detectors) array is written to --output as dataset "
            "/snr with catalog_filename and detectors attributes."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help=(
            "Path to the catalog .h5 file. Only its /source_parameters group is "
            "read; it must contain tc, ra, dec, psi (no values are drawn)."
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
        "--waveform-model",
        default=None,
        help=(
            "LAL approximant name. Defaults to the catalog's 'approximant' "
            "root attribute when present."
        ),
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        default=None,
        help=(
            "Sample rate in Hz; sets the Nyquist frequency. Defaults to the "
            "catalog's 'sampling_frequency' root attribute when present, "
            "otherwise 2048 Hz."
        ),
    )
    parser.add_argument(
        "--minimum-frequency",
        type=float,
        default=None,
        help=(
            "Lower bound of the SNR integral and waveform-generation cutoff in "
            "Hz. Defaults to the catalog's 'minimum_frequency' root attribute "
            "when present, otherwise 20 Hz."
        ),
    )
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        default=None,
        help="Optional upper bound of the SNR integral in Hz; defaults to Nyquist.",
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


def load_source_parameters(
    catalog_path: Path,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
    """Read only the /source_parameters group of the catalog.

    Returns the per-event parameter arrays and the catalog's root attributes
    (for frequency/approximant defaults). Pre-computed polarization datasets
    are never opened.
    """
    with h5py.File(catalog_path, "r") as catalog:
        if "source_parameters" not in catalog:
            raise SystemExit(
                f"{catalog_path} has no /source_parameters group; not a "
                "catalog compatible with compute-snr."
            )
        group = catalog["source_parameters"]
        source_parameters = {
            key: np.asarray(group[key][()], dtype=np.float64) for key in group
        }
        # Attributes must be copied out: they die with the closed file handle.
        root_attributes = dict(catalog.attrs)

    missing = [
        key for key in REQUIRED_SOURCE_PARAMETERS if key not in source_parameters
    ]
    if missing:
        raise SystemExit(
            f"{catalog_path} is missing required source parameters "
            f"{missing}: this CLI does not draw sky position or polarization "
            "angle, so the catalog itself must carry tc, ra, dec, and psi."
        )
    return source_parameters, root_attributes


def _progress(done: int, total: int) -> None:
    if done == total or done == 1 or done % PROGRESS_LOG_EVERY == 0:
        logger.info("Computed SNR for %d/%d events", done, total)


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

    source_parameters, catalog_attributes = load_source_parameters(catalog_path)
    n_events = len(source_parameters[REQUIRED_SOURCE_PARAMETERS[0]])
    logger.info("Loaded %d events from %s", n_events, catalog_path)

    waveform_model = args.waveform_model or catalog_attributes.get("approximant")
    if waveform_model is None:
        raise SystemExit(
            "No waveform model: pass --waveform-model (the catalog has no "
            "'approximant' attribute)."
        )
    sampling_frequency = float(
        args.sampling_frequency or catalog_attributes.get("sampling_frequency", 2048.0)
    )
    minimum_frequency = float(
        args.minimum_frequency or catalog_attributes.get("minimum_frequency", 20.0)
    )
    logger.info(
        "Waveform model %s, sampling %.1f Hz, band [%.1f, %s] Hz",
        waveform_model,
        sampling_frequency,
        minimum_frequency,
        args.maximum_frequency if args.maximum_frequency is not None else "Nyquist",
    )

    snrs = optimal_snr(
        source_parameters,
        detectors,
        sensitivities,
        waveform_model=str(waveform_model),
        sampling_frequency=sampling_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=args.maximum_frequency,
        earth_rotation=not args.no_earth_rotation,
        progress_callback=_progress,
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
