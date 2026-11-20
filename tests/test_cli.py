"""CLI runtime initialization."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import tomli_w
from astrogwb_tiny_pipeline import WAVEFORM, write_population_csv
from typer.testing import CliRunner

from astrogwb.background.simulation import BackgroundDraw
from astrogwb.cli import app
from astrogwb.detector import load_detector, network_pairs
from astrogwb.utils import years_to_seconds


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
def inputs(tmp_path: Path) -> dict[str, Path]:
    """A gwmock-pop catalogue, the waveform settings and a two-detector network."""
    waveform = tmp_path / "waveform.toml"
    waveform.write_bytes(
        tomli_w.dumps(WAVEFORM.model_dump(mode="json", exclude_none=True)).encode()
    )
    detector = tmp_path / "detector.toml"
    names = ["S1", "R1"]
    detectors = {
        name: load_detector(name).model_dump(mode="json", exclude_none=True)
        for name in names
    }
    detector.write_bytes(
        tomli_w.dumps({"detectors": detectors, "networks": {"ET": names}}).encode()
    )
    return {
        "population": write_population_csv(tmp_path / "events.csv"),
        "waveform": waveform,
        "detector": detector,
    }


def _simulate(output: Path, inputs: dict[str, Path], *options: str) -> None:
    result = CliRunner().invoke(
        app,
        [
            "simulate",
            str(output),
            str(inputs["population"]),
            *("--waveform-config", str(inputs["waveform"])),
            *("--detector-config", str(inputs["detector"])),
            *("--rate-per-year", str(8.0 * years_to_seconds(1.0))),
            *("--num-segments", "2"),
            *("--segment-seconds", "1"),
            *("--seed", "41"),
            *("--chunk-size", "8"),
            *options,
        ],
    )
    assert result.exit_code == 0, result.output


def test_a_simulate_run_writes_one_file_with_its_config(
    tmp_path: Path, inputs: dict[str, Path]
) -> None:
    output = tmp_path / "41-0.h5"

    _simulate(output, inputs)

    draw, config = BackgroundDraw.read(output)
    assert tuple(config.networks) == ("ET",)
    assert network_pairs(config.networks["ET"]) == (("R1", "S1"),)
    assert config.population == str(inputs["population"])
    assert (config.seed, config.index, config.incoherent) == (41, 0, False)
    assert config.observer_rate_hz == pytest.approx(8.0)
    assert draw.segment_counts.shape == (2,)
    assert draw.segment_counts.sum() > 0
    assert np.any(draw.spectra["ET"])
    with h5py.File(output, "r") as file:
        metadata = json.loads(file.attrs["metadata"])
    assert set(metadata["detectors"]["detectors"]) == {"R1", "S1"}
    assert [path.name for path in tmp_path.glob("*.h5")] == ["41-0.h5"]


def test_a_simulate_run_with_incoherent_records_the_mode(
    tmp_path: Path, inputs: dict[str, Path]
) -> None:
    output = tmp_path / "41-0.h5"

    _simulate(output, inputs, "--incoherent")

    assert BackgroundDraw.read(output)[1].incoherent
