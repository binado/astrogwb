from __future__ import annotations

import argparse
import logging
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import xarray as xr
from astrogwb.detector import load_sensitivity_map, resolve_detector
from astrogwb.resolved import optimal_snr
from astrogwb.waveform import open_catalog, save_catalog

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
            "Load an astrogwb waveform catalog, generate fresh time-domain "
            "waveforms for every event using its waveform metadata, project "
            "them onto every requested detector, and compute per-detector optimal "
            "SNR. The catalog is enriched with snr(sample, detector), sorted by "
            "descending all-detector network SNR, and atomically replaced in place."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help=("Path to the astrogwb waveform catalog to enrich and replace in place."),
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
        "--batch-size",
        type=_positive_int,
        required=True,
        help=(
            "Maximum number of duration-sorted events sharing one waveform grid. "
            "Ripple may reduce a batch further to respect device memory."
        ),
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


def _source_parameters(catalog: xr.Dataset) -> dict[str, np.ndarray]:
    """Unstack the catalog's sample parameters without loading waveform power."""
    return {
        str(name): np.asarray(
            catalog.source_parameters.sel(parameter=name).values, dtype=np.float64
        )
        for name in catalog.parameter.values
    }


def _without_old_snr(catalog: xr.Dataset) -> xr.Dataset:
    if "detector" in catalog.dims:
        return catalog.drop_dims("detector")
    if "snr" in catalog:
        return catalog.drop_vars("snr")
    return catalog


def _temporary_path(destination: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(raw_path)


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

    temporary_path: Path | None = None
    try:
        with open_catalog(catalog_path) as catalog:
            if not catalog.attrs.get("approximant"):
                raise SystemExit("Catalog approximant must not be empty.")
            sampling_frequency = float(catalog.attrs["sampling_frequency"])
            minimum_frequency = float(catalog.attrs["minimum_frequency"])
            maximum_frequency = float(catalog.attrs["maximum_frequency"])
            if sampling_frequency <= 0:
                raise SystemExit("Catalog sampling_frequency must be positive.")
            if minimum_frequency <= 0:
                raise SystemExit("Catalog minimum_frequency must be positive.")
            if maximum_frequency <= minimum_frequency:
                raise SystemExit(
                    "Catalog maximum_frequency must be greater than minimum_frequency."
                )

            logger.info(
                "Loaded %d events from %s", catalog.sizes["sample"], catalog_path
            )
            logger.info(
                "Waveform model %s, sampling %.1f Hz, band [%.1f, %.1f] Hz",
                catalog.attrs["approximant"],
                sampling_frequency,
                minimum_frequency,
                maximum_frequency,
            )

            earth_rotation = not args.no_earth_rotation
            snrs = optimal_snr(
                _source_parameters(catalog),
                detectors,
                sensitivities,
                waveform_model=str(catalog.attrs["approximant"]),
                sampling_frequency=sampling_frequency,
                minimum_frequency=minimum_frequency,
                batch_size=args.batch_size,
                maximum_frequency=maximum_frequency,
                earth_rotation=earth_rotation,
                backend=args.backend,
                progress_callback=_progress_callback(args.progress_log_every),
            )
            network_snr = np.sqrt(np.sum(snrs**2, axis=1))
            order = np.argsort(-network_snr, kind="stable")

            enriched = _without_old_snr(catalog).assign_coords(
                detector=np.asarray(detector_names, dtype=str)
            )
            enriched = enriched.assign(snr=(("sample", "detector"), snrs))
            enriched = enriched.isel(sample=order)
            enriched.attrs.update(
                {
                    "snr_backend": args.backend,
                    "snr_earth_rotation": int(earth_rotation),
                    "snr_batch_size": args.batch_size,
                }
            )

            temporary_path = _temporary_path(catalog_path)
            save_catalog(temporary_path, enriched)
            with open_catalog(temporary_path):
                pass
            os.chmod(temporary_path, stat.S_IMODE(catalog_path.stat().st_mode))

        os.replace(temporary_path, catalog_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    logger.info(
        "Stored SNR for %d detectors and reordered %s in place",
        len(detector_names),
        catalog_path,
    )


if __name__ == "__main__":
    main()
