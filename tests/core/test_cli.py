"""The ``astrogwb simulate`` command line."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from astrogwb_tiny_pipeline import WAVEFORM, population_metadata
from typer.testing import CliRunner

from astrogwb.cli import app
from astrogwb.simulators import Realization

POPULATION = population_metadata()


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


@pytest.fixture
def metadata_files(tmp_path: Path) -> dict[str, Path]:
    """The tiny population, waveform and two-detector network, as TOML files."""
    tables = {
        "population": POPULATION.model_dump(mode="json", exclude={"version"}),
        "waveform": WAVEFORM.model_dump(mode="json", exclude_none=True),
        "network": {
            "detectors": ["S1", "R1"],
            "psd_references": {
                "S1": "ET_COBA_15km_psd.txt",
                "R1": "ET_COBA_15km_psd.txt",
            },
        },
    }
    paths = {name: tmp_path / f"{name}.toml" for name in tables}
    for name, table in tables.items():
        paths[name].write_bytes(tomli_w.dumps(table).encode())
    return paths


def _simulate(directory: Path, *options: str) -> None:
    result = CliRunner().invoke(
        app, ["simulate", str(directory), "--chunk-size", "8", *options]
    )
    assert result.exit_code == 0, result.output


def test_a_fused_simulate_run_writes_the_manifest_sources_and_background(
    tmp_path: Path, metadata_files: dict[str, Path]
) -> None:
    _simulate(
        tmp_path / "realization",
        *("--population-metadata", str(metadata_files["population"])),
        *("--seed", "41"),
        *("--waveform-metadata", str(metadata_files["waveform"])),
        *("--background-metadata", str(metadata_files["network"])),
    )

    realization = Realization.open(tmp_path / "realization")
    background = realization.background()
    assert background.pairs == [("S1", "R1")]
    assert background.count > 0 and np.any(background.pair_spectral_density)
    assert sorted(path.name for path in realization.directory.iterdir()) == [
        "background.h5",
        "metadata.toml",
        "source_parameters.h5",
    ]


def test_a_background_run_adds_to_an_existing_realization(
    tmp_path: Path, metadata_files: dict[str, Path]
) -> None:
    directory = tmp_path / "realization"
    _simulate(
        directory,
        *("--population-metadata", str(metadata_files["population"])),
        *("--seed", "41"),
        *("--waveform-metadata", str(metadata_files["waveform"])),
    )
    assert not (directory / "background.h5").exists()

    _simulate(directory, "--background-metadata", str(metadata_files["network"]))

    assert Realization.open(directory).background().pairs == [("S1", "R1")]


def test_a_seed_without_population_metadata_is_refused(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["simulate", str(tmp_path), "--seed", "41"])

    assert result.exit_code != 0
    assert "go together" in result.output
