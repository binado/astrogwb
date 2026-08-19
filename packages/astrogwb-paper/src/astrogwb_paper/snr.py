"""Fiducial matched-filter SNR of the injected background, per detector network.

Deduplicated from two byte-identical copies in the chain-plotting scripts. It
lives here rather than in :mod:`astrogwb_paper.inference` because it returns a
``pandas.DataFrame`` and pandas is only in the ``plotting`` dependency group,
not a runtime dependency; and not in :mod:`astrogwb_paper.plotting`, which is
documented as presentation-only and imports nothing from ``astrogwb``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astrogwb_paper.config.analysis import AnalysisGrid
from astrogwb_paper.inference import prepare_observation

if TYPE_CHECKING:
    import pandas as pd

    from astrogwb_paper.config.figures import Network


def compute_network_snrs(
    catalog_path: Path,
    networks: Sequence[Network],
    fiducials: Mapping[str, float],
    *,
    grid: AnalysisGrid,
    jnp: Any,
) -> pd.DataFrame:
    """Compute the fiducial matched-filter SNR for each detector network."""
    import pandas as pd
    from astrogwb.detector import effective_psd, load_sensitivity_map
    from astrogwb.frequency import frequency_spacing as compute_frequency_spacing
    from astrogwb.gwb import spectral_snr
    from astrogwb.utils import years_to_seconds

    observation = prepare_observation(
        catalog_path, fiducials=fiducials, grid=grid, jnp=jnp
    )
    frequencies = observation.frequencies
    mask = observation.frequency_mask
    observed_spectral_density = observation.spectral_density
    frequency_spacing = compute_frequency_spacing(frequencies)
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
                observed_spectral_density[mask],
                effective_noise[mask],
                observation_seconds,
                frequency_spacing,
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
