"""The ``astrogwb simulate`` command line."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import tomli_w
from typer.testing import CliRunner

from astrogwb.cli import app
from astrogwb.simulators import BackgroundArtifact


def test_importing_the_cli_leaves_the_xla_backend_uninitialized() -> None:
    """A late ``set_host_device_count(2)`` still taking effect proves it."""
    code = """
from typer.testing import CliRunner
from astrogwb.cli import app

assert CliRunner().invoke(app, ["simulate", "--help"]).exit_code == 0
import numpyro

numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"backend initialized early: {jax.devices()}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_a_fused_background_run_writes_only_the_background(tmp_path: Path) -> None:
    metadata = {
        "population.toml": {
            "observation_time": 1e-4,
            "population": {
                "model_name": "bns_coba",
                "model_kwargs": {
                    "mass_model": "uniform",
                    "minimum_redshift": 0.0,
                    "maximum_redshift": 5.0,
                    "n_grid": 64,
                },
            },
            "hyperparameters": {
                "H0": 67.66,
                "Omega_m": 0.3096,
                "gamma": 1.42,
                "kappa": 4.62,
                "z_peak": 1.84,
                "minimum_mass": 1.0,
                "mass_width": 1.5,
                "local_merger_rate": 750.0,
            },
        },
        "waveform.toml": {
            "approximant": "AnalyticInspiral",
            "minimum_frequency": 20.0,
            "maximum_frequency": 50.0,
            "reference_frequency": 20.0,
            "sampling_frequency": 128.0,
            "frequency_resolution": 10.0,
        },
        "network.toml": {
            "detectors": ["S1", "R1"],
            "psd_references": {
                "S1": "ET_COBA_15km_psd.txt",
                "R1": "ET_COBA_15km_psd.txt",
            },
        },
    }
    for name, table in metadata.items():
        (tmp_path / name).write_bytes(tomli_w.dumps(table).encode())

    result = CliRunner().invoke(
        app,
        [
            "simulate",
            "background",
            "--population-metadata",
            str(tmp_path / "population.toml"),
            "--seed",
            "41",
            "--waveform-metadata",
            str(tmp_path / "waveform.toml"),
            "--detectors",
            str(tmp_path / "network.toml"),
            "--output",
            str(tmp_path / "background"),
            "--chunk-size",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    background = BackgroundArtifact.open(tmp_path / "background")
    assert background.pairs == [("S1", "R1")]
    assert background.count > 0 and np.any(background.pair_spectral_density)
    assert sorted(path.name for path in (tmp_path / "background").iterdir()) == [
        "data.h5",
        "metadata.toml",
    ]
