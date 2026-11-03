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
from astrogwb.simulators import BackgroundArtifact

POPULATION = population_metadata()


def test_importing_the_cli_leaves_the_xla_backend_uninitialized() -> None:
    """A late ``set_host_device_count(2)`` still taking effect proves it."""
    command = "simulate"
    code = f"""
from typer.testing import CliRunner
from astrogwb.cli import app

assert CliRunner().invoke(app, ["{command}", "--help"]).exit_code == 0
import numpyro

numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"backend initialized early: {{jax.devices()}}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


@pytest.fixture
def metadata_files(tmp_path: Path) -> dict[str, Path]:
    """The tiny population, waveform and two-detector network, as TOML files."""
    tables = {
        "population": POPULATION.model_dump(
            mode="json", exclude={"version"}, exclude_none=True
        ),
        "waveform": WAVEFORM.model_dump(mode="json", exclude_none=True),
        "detector": {"networks": {"ET": ["S1", "R1"]}},
    }
    paths = {name: tmp_path / f"{name}.toml" for name in tables}
    for name, table in tables.items():
        paths[name].write_bytes(tomli_w.dumps(table).encode())
    return paths


def _simulate(output: Path, *options: str) -> None:
    result = CliRunner().invoke(
        app, ["simulate", str(output), "--chunk-size", "8", *options]
    )
    assert result.exit_code == 0, result.output


@pytest.fixture
def required(metadata_files: dict[str, Path]) -> list[str]:
    """Every required option of ``simulate``, but the output."""
    return [
        *("--population-metadata", str(metadata_files["population"])),
        *("--waveform-metadata", str(metadata_files["waveform"])),
        *("--detector-config", str(metadata_files["detector"])),
        *("--seed", "41"),
    ]


def test_a_simulate_run_writes_one_file(tmp_path: Path, required: list[str]) -> None:
    output = tmp_path / "41-0.h5"

    _simulate(output, *required)

    background = BackgroundArtifact.open(output)
    assert background.pairs == {"ET": (("R1", "S1"),)}
    assert background.count > 0
    assert background.segment_duration is None
    assert np.any(background.spectra["ET"])
    assert [path.name for path in tmp_path.glob("*.h5")] == ["41-0.h5"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "41-0.h5",
        "detector.toml",
        "population.toml",
        "waveform.toml",
    ]


def test_segment_duration_is_recorded_with_the_noise(
    tmp_path: Path, required: list[str]
) -> None:
    output = tmp_path / "41-0.h5"

    _simulate(output, *required, "--segment-duration", "600")

    background = BackgroundArtifact.open(output)
    assert background.segment_duration == 600.0
    noise = background.noise()
    assert noise["ET"].pair_scale.shape == (1, background.frequencies.size)
