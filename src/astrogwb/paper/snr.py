"""Fiducial matched-filter SNR of the injected background, per detector network.

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

from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.gwb import spectral_snr
from astrogwb.paper.config.mcmc import AnalysisGrid
from astrogwb.paper.inference import prepare_observation
from astrogwb.utils import years_to_seconds

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr

    from astrogwb.paper.plotting import Network


def compute_network_snrs(
    injection_catalog: Path | xr.Dataset,
    networks: Sequence[Network],
    fiducials: Mapping[str, float],
    *,
    grid: AnalysisGrid,
) -> pd.DataFrame:
    """Compute the fiducial matched-filter SNR for each detector network.

    ``injection_catalog`` is the injection catalog the run's chains were
    sampled against, either a path to load or an already-loaded dataset (a
    notebook generating its catalog in-process passes one directly). A path
    is resolved from the same merged run config that supplied ``fiducials``
    and ``grid``, so a figure's SNR is computed over exactly that catalog.
    """
    import pandas as pd
    import xarray as xr

    from astrogwb.paper.catalogs import load_run_catalog

    catalog = (
        injection_catalog
        if isinstance(injection_catalog, xr.Dataset)
        else load_run_catalog(injection_catalog, label="injection")
    )
    observation = prepare_observation(catalog, fiducials=fiducials, grid=grid)
    frequencies = observation.frequencies
    band = observation.frequency_mask
    observed_spectral_density = observation.spectral_density
    observation_seconds = years_to_seconds(grid.observation_time)

    rows: list[dict[str, Any]] = []
    for network in networks:
        detectors = network.detectors
        sensitivities = load_sensitivity_map(detectors)
        effective_noise = jnp.asarray(
            effective_psd(frequencies, list(detectors), sensitivities)
        )
        snr = float(
            spectral_snr(
                observed_spectral_density[band],
                effective_noise[band],
                observation_seconds,
                observation.df,
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
