"""Ensembles of network realizations, read from an explicit ensemble root.

An ensemble is a directory of realizations ``<root>/<seed>-<index>``, each
written by ``astrogwb simulate`` with every paper network in its background,
and ``<root>/detector_noise.h5`` written by ``astrogwb noise``. Generating them
is a separate, explicit step (see ``workflow/coverage_draws.smk``);
:func:`network_spectra_ensemble` only reads, and a missing draw is a
:class:`FileNotFoundError`, never a simulation.

The simulations' inputs are committed TOML files written by
``scripts/write_ensemble_configs.py`` from :func:`paper_networks` and
:func:`ensemble_metadata`: ``config/networks.toml`` and
``config/ensembles/<name>/{population,waveform}.toml``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from astrogwb.paper.config import (
    detector_registry,
    fiducials,
    population_metadata,
    waveform_metadata,
)
from astrogwb.paper.config.detectors import DetectorRegistry
from astrogwb.simulators import (
    BackgroundArtifact,
    DetectorConfiguration,
    PopulationMetadata,
    Realization,
)
from astrogwb.waveform import WaveformMetadata

__all__ = [
    "ENSEMBLES",
    "ETD_NETWORKS",
    "ETD_PSD",
    "NetworkSpectra",
    "ensemble_config_tables",
    "ensemble_metadata",
    "network_configuration",
    "network_spectra_ensemble",
    "paper_networks",
]

#: The ET-D noise curve, weighting the ET triangle's pairs in the ET-D networks.
ETD_PSD = "ET-000A-18_ETD_hlf_psd.txt"

#: Each ET-triangle network and its ET-D counterpart: same pairs, other weights.
ETD_NETWORKS = {
    "ET-triangular": "ET-triangular-D-PSD",
    "ET-triangular-CE-Hanford": "ET-triangular-D-PSD-CE-Hanford",
}

#: Observation time of a smoke ensemble, in years: a few hundred sources.
SMOKE_OBSERVATION_TIME = 1e-4

#: The waveform overrides of each ensemble, over the shared ``[catalog]`` one.
#: ``fiducial`` is the injection of the grid notebooks and the Poisson draws of
#: ``spectrum_snrs``; ``coverage`` the draws of ``shot_noise_coverage`` and
#: ``ensemble_spectrum``, fine through the SNR peak with a log tail.
_WAVEFORMS: dict[str, dict[str, object]] = {
    "fiducial": {},
    "coverage": {
        "approximant": "IMRPhenomXAS_NRTidalv3",
        "frequency_resolution": 0.125,
        "turnover_frequency": 100.0,
        "dlogf": 0.01266,
    },
}

#: Every ensemble name; ``<name>-smoke`` observes for :data:`SMOKE_OBSERVATION_TIME`.
ENSEMBLES = tuple(name for base in _WAVEFORMS for name in (base, f"{base}-smoke"))


class NetworkSpectra(TypedDict):
    """One network's draws, stacked along the leading draw axis.

    ``frequencies`` ``(F,)``; ``spectral_density`` ``(D, F)``, the
    :func:`~astrogwb.simulators.combine_pair_spectra` of
    ``pair_spectral_density`` ``(D, P, F)``; ``n_events`` and
    ``total_merger_rate`` ``(D,)``; each ``hyperparameters`` value ``(D,)``.
    """

    frequencies: NDArray[np.float64]
    spectral_density: NDArray[np.float64]
    pair_spectral_density: NDArray[np.float64]
    n_events: NDArray[np.int64]
    total_merger_rate: NDArray[np.float64]
    hyperparameters: dict[str, NDArray[np.float64]]


def network_configuration(
    registry: DetectorRegistry, name: str
) -> DetectorConfiguration:
    """The network ``name`` of ``registry``, with its noise curves.

    Detectors are named by site code, which resolves to the packaged geometry:
    a registry geometry override is not carried into it.
    """
    detectors = list(registry.networks[name])
    return DetectorConfiguration(
        detectors=detectors,
        psd_references={
            detector: registry.detectors[detector].psd_reference
            for detector in detectors
        },
    )


def network_spectra_ensemble(
    root: str | Path,
    seed: int,
    num_draws: int,
    names: Sequence[str],
    *,
    start: int = 0,
) -> list[NetworkSpectra]:
    """Draws ``start .. num_draws - 1`` of the networks ``names``, stacked per network.

    Draw ``i`` is the background of the realization at ``<root>/<seed>-<i>``.

    Parameters
    ----------
    root
        The ensemble root.
    seed
        The seed the ensemble was simulated with.
    num_draws
        One past the last index read.
    names
        Networks, each recorded in every draw's background.
    start
        The first index read.

    Returns
    -------
    list of NetworkSpectra
        Each network's draws, in ``names`` order.

    Raises
    ------
    FileNotFoundError
        If a draw, or its background, is missing.
    KeyError
        If a draw's background does not record one of ``names``.
    """
    root = Path(root)
    draws = [
        Realization.open(root / f"{int(seed)}-{index}").background()
        for index in range(start, num_draws)
    ]
    return [_stack(draws, name) for name in names]


def _stack(draws: Sequence[BackgroundArtifact], name: str) -> NetworkSpectra:
    """One network's spectra of every draw, stacked."""
    spectra = [draw.network(name) for draw in draws]
    return {
        "frequencies": draws[0].frequencies,
        "spectral_density": np.stack([s.spectral_density for s in spectra]),
        "pair_spectral_density": np.stack([s.pair_spectral_density for s in spectra]),
        "n_events": np.asarray([draw.count for draw in draws], dtype=np.int64),
        "total_merger_rate": np.asarray(
            [draw.total_merger_rate for draw in draws], dtype=np.float64
        ),
        "hyperparameters": {
            key: np.asarray([draw.hyperparameters[key] for draw in draws])
            for key in draws[0].hyperparameters
        },
    }


def paper_networks(registry: DetectorRegistry) -> dict[str, DetectorConfiguration]:
    """Every network of ``registry``, and the :data:`ETD_NETWORKS` variants."""
    networks = {
        name: network_configuration(registry, name) for name in registry.networks
    }
    for base, swapped in ETD_NETWORKS.items():
        references = networks[base].psd_references
        networks[swapped] = DetectorConfiguration(
            detectors=networks[base].detectors,
            psd_references={
                detector: ETD_PSD if detector.startswith("E") else reference
                for detector, reference in references.items()
            },
        )
    return networks


def ensemble_metadata(
    name: str, root: Path | None = None
) -> tuple[PopulationMetadata, WaveformMetadata]:
    """The population and waveform settings of the ensemble ``name``.

    The population is the shared ``[catalog]`` one at the fiducials, observed
    for a year, or :data:`SMOKE_OBSERVATION_TIME` for a ``-smoke`` ensemble.

    Raises
    ------
    KeyError
        If ``name`` is not one of :data:`ENSEMBLES`.
    """
    base = name.removesuffix("-smoke")
    waveform = waveform_metadata(root, **_WAVEFORMS[base])
    population = PopulationMetadata(
        observation_time=SMOKE_OBSERVATION_TIME if name.endswith("-smoke") else 1.0,
        hyperparameters={**fiducials(root)},
        population=population_metadata(root),
    )
    return population, waveform


def ensemble_config_tables(root: Path | None = None) -> dict[Path, dict[str, object]]:
    """The committed simulation inputs, by path relative to ``root``, as TOML tables.

    ``config/networks.toml`` holds :func:`paper_networks` as
    ``[networks.<name>]`` tables; ``config/ensembles/<name>/population.toml``
    and ``waveform.toml`` hold :func:`ensemble_metadata` of every
    :data:`ENSEMBLES` name.
    """
    networks = paper_networks(detector_registry(root=root))
    tables: dict[Path, dict[str, object]] = {
        Path("config/networks.toml"): {
            "networks": {
                name: network.model_dump(mode="json")
                for name, network in networks.items()
            }
        }
    }
    for name in ENSEMBLES:
        population, waveform = ensemble_metadata(name, root)
        directory = Path("config/ensembles") / name
        tables[directory / "population.toml"] = population.model_dump(
            mode="json", exclude={"version"}, exclude_none=True
        )
        tables[directory / "waveform.toml"] = waveform.model_dump(
            mode="json", exclude_none=True
        )
    return tables
