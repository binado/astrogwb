"""Matched-filter SNRs of injection catalogs and spectrum ensembles.

Deduplicated from two byte-identical copies in the chain-plotting scripts. It
lives here rather than in :mod:`astrogwb.paper.inference` because it returns a
``pandas.DataFrame`` and pandas is only in the ``plotting`` dependency group,
not a runtime dependency; and not in :mod:`astrogwb.paper.plotting`, which is
documented as presentation-only and imports nothing from ``astrogwb``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.frequency import frequency_mask
from astrogwb.gwb import spectral_snr
from astrogwb.paper.inference import prepare_observation
from astrogwb.utils import years_to_seconds

if TYPE_CHECKING:
    import pandas as pd

    from astrogwb.catalog import SpectralDensityCatalog
    from astrogwb.paper.config.detectors import DetectorRegistry
    from astrogwb.paper.plotting import Network


def compute_spectrum_snrs(
    catalog: SpectralDensityCatalog,
    detector_registry: DetectorRegistry,
    network: str,
    *,
    minimum_frequency: float,
    maximum_frequency: float,
) -> tuple[NDArray[np.float64], float]:
    """Return per-realization SNRs and the mean spectrum's SNR for one network.

    The observing time comes from the artifact, in years. Both calculations
    use the full frequency grid and mask afterwards, preserving the widths
    of bins at the analysis band's edges and all within-row correlations.
    No source sampling or waveform generation is performed here.
    """
    if not (
        np.isfinite(minimum_frequency)
        and np.isfinite(maximum_frequency)
        and 0.0 <= minimum_frequency < maximum_frequency
    ):
        raise ValueError("frequency bounds must be finite and 0 <= minimum < maximum")
    if not np.isfinite(catalog.observation_time) or catalog.observation_time <= 0:
        raise ValueError("observation_time must be finite and positive")
    if catalog.frequencies.size < 2:
        raise ValueError("SNR calculation requires at least two frequency bins")
    if network not in detector_registry.networks:
        raise ValueError(f"unknown network {network!r}")

    frequencies = jnp.asarray(catalog.frequencies)
    band = frequency_mask(frequencies, fmin=minimum_frequency, fmax=maximum_frequency)
    if not np.any(np.asarray(band)):
        raise ValueError("the analysis frequency band contains no spectrum bins")
    geometry, sensitivities = detector_registry.build_network(network)
    noise = jnp.asarray(effective_psd(frequencies, geometry, sensitivities))
    spectra = jnp.asarray(catalog.spectral_density)
    seconds = years_to_seconds(catalog.observation_time)
    snrs = np.asarray(
        spectral_snr(spectra, noise, seconds, frequencies, frequency_mask=band),
        dtype=np.float64,
    )
    mean_spectrum_snr = float(
        spectral_snr(
            jnp.mean(spectra, axis=0),
            noise,
            seconds,
            frequencies,
            frequency_mask=band,
        )
    )
    if not np.all(np.isfinite(snrs)) or not np.isfinite(mean_spectrum_snr):
        raise ValueError("SNR calculation produced nonfinite results")
    return snrs, mean_spectrum_snr


def summarize_spectrum_snrs(
    snrs: ArrayLike, *, mean_spectrum_snr: float
) -> dict[str, int | float]:
    """Summarize realization SNRs, keeping the mean-spectrum statistic separate.

    A single draw has undefined sample SD; relative scatter is undefined when
    the mean is zero or SD is undefined. These values are returned as NaN,
    which table writers can represent as missing values.
    """
    values = np.asarray(snrs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("snrs must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("snrs must be finite and nonnegative")
    if not np.isfinite(mean_spectrum_snr) or mean_spectrum_snr < 0:
        raise ValueError("mean_spectrum_snr must be finite and nonnegative")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
    return {
        "num_draws": int(values.size),
        "mean": mean,
        "median": float(np.median(values)),
        "sd": sd,
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "relative_scatter": sd / mean if mean > 0 else float("nan"),
        "mean_spectrum_snr": mean_spectrum_snr,
    }


def compute_network_snrs(
    injection_catalog_path: Path,
    networks: Sequence[Network],
    fiducials: Mapping[str, float],
    *,
    observation_time: float,
    minimum_redshift: float,
    maximum_redshift: float,
    minimum_frequency: float,
    maximum_frequency: float,
) -> pd.DataFrame:
    """Compute the fiducial matched-filter SNR for each detector network.

    ``injection_catalog_path`` is the injection catalog file the run's chains
    were sampled against; the caller resolves it from the same merged run
    config that supplied ``fiducials`` and ``grid``, so a figure's SNR is
    computed over exactly that catalog.
    """
    import pandas as pd

    from astrogwb.paper.catalogs import load_run_catalog

    catalog = load_run_catalog(injection_catalog_path, label="injection")
    observation = prepare_observation(
        catalog,
        minimum_redshift=minimum_redshift,
        maximum_redshift=maximum_redshift,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    frequencies = observation.frequencies
    band = observation.frequency_mask
    observed_spectral_density = observation.spectral_density
    observation_seconds = years_to_seconds(observation_time)

    rows: list[dict[str, Any]] = []
    for network in networks:
        detectors = network.detectors
        if network.detector_registry is None:
            geometry = list(detectors)
            sensitivities = load_sensitivity_map(detectors)
        else:
            geometry, sensitivities = network.detector_registry.build_detectors(
                detectors
            )
        effective_noise = jnp.asarray(
            effective_psd(frequencies, geometry, sensitivities)
        )
        # Full-grid arrays plus the mask, not `[band]` slices: bin widths are
        # derived from the whole axis, so slicing first would mis-size the
        # bins at the band's edges.
        snr = float(
            spectral_snr(
                observed_spectral_density,
                effective_noise,
                observation_seconds,
                frequencies,
                frequency_mask=band,
            )
        )
        rows.append(
            {
                "network": network.name,
                "detectors": ",".join(detectors),
                "n_detectors": len(detectors),
                "snr": snr,
            }
        )
    return pd.DataFrame(rows)
