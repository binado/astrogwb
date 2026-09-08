"""Tests for astrogwb.paper.snr.compute_network_snrs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from catalog_fixtures import make_catalog, save_catalog
from config_fixtures import example_raw

from astrogwb.paper.config.mcmc import build_run_config
from astrogwb.paper.plotting import Network
from astrogwb.paper.snr import compute_network_snrs

pytestmark = pytest.mark.integration

# Same band and redshift window as test_inference_pipeline.py's fixtures: a
# real ET+CE network (S1, R1, C1) has finite, positive effective PSD there.
FREQUENCIES = np.linspace(10.0, 50.0, 5)
N_SOURCES = 8


def _injection_catalog(*, seed: int):
    rng = np.random.default_rng(seed)
    redshift = np.linspace(0.05, 1.5, N_SOURCES)
    source_parameters = {
        "redshift": redshift,
        "luminosity_distance": 1.0e3 * (1.0 + redshift),
        "mass_1": np.full(N_SOURCES, 1.4),
        "mass_2": np.full(N_SOURCES, 1.4),
    }
    return make_catalog(
        frequencies=FREQUENCIES,
        polarization_power=rng.uniform(0.0, 1.0, size=(FREQUENCIES.size, N_SOURCES)),
        source_parameters=source_parameters,
        approximant="Toy",
        minimum_frequency=float(FREQUENCIES[0]),
        maximum_frequency=float(FREQUENCIES[-1]),
        reference_frequency=20.0,
        sampling_frequency=128.0,
        df=float(FREQUENCIES[1] - FREQUENCIES[0]),
    )


def test_compute_network_snrs_dataset_matches_path(tmp_path: Path) -> None:
    config = build_run_config(example_raw())
    catalog = _injection_catalog(seed=0)
    path = tmp_path / "injection.h5"
    save_catalog(path, catalog)
    network = Network("test-network", "test", config.analysis.detectors)

    from_path = compute_network_snrs(
        path, [network], config.fiducials, grid=config.analysis_grid
    )
    from_dataset = compute_network_snrs(
        catalog, [network], config.fiducials, grid=config.analysis_grid
    )

    np.testing.assert_array_equal(from_path["snr"], from_dataset["snr"])
