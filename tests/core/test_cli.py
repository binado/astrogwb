"""The ``astrogwb simulate`` and ``astrogwb noise`` command lines."""

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
from astrogwb.simulators import DetectorNoise, Realization

POPULATION = population_metadata()


@pytest.mark.parametrize("command", ["simulate", "noise"])
def test_importing_the_cli_leaves_the_xla_backend_uninitialized(command: str) -> None:
    """A late ``set_host_device_count(2)`` still taking effect proves it."""
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
        "population": POPULATION.model_dump(mode="json", exclude={"version"}),
        "waveform": WAVEFORM.model_dump(mode="json", exclude_none=True),
        "detector": {"networks": {"ET": ["S1", "R1"]}},
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
        *("--detector-config", str(metadata_files["detector"])),
    )

    realization = Realization.open(tmp_path / "realization")
    background = realization.background()
    assert background.pairs == {"ET": (("R1", "S1"),)}
    assert background.count > 0
    assert np.any(background.network("ET").pair_spectral_density)
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

    _simulate(directory, "--detector-config", str(metadata_files["detector"]))

    assert Realization.open(directory).background().network_names == ("ET",)


def test_a_seed_without_population_metadata_is_refused(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["simulate", str(tmp_path), "--seed", "41"])

    assert result.exit_code != 0
    assert "go together" in result.output


def test_noise_writes_the_backgrounds_networks_on_its_grid(
    tmp_path: Path, metadata_files: dict[str, Path]
) -> None:
    directory = tmp_path / "41-0"
    _simulate(
        directory,
        *("--population-metadata", str(metadata_files["population"])),
        *("--seed", "41"),
        *("--waveform-metadata", str(metadata_files["waveform"])),
        *("--detector-config", str(metadata_files["detector"])),
    )

    result = CliRunner().invoke(
        app,
        [
            "noise",
            str(directory),
            str(tmp_path / "detector_noise.h5"),
            "--detector-config",
            str(metadata_files["detector"]),
        ],
    )

    assert result.exit_code == 0, result.output
    noise = DetectorNoise.open(tmp_path / "detector_noise.h5")
    realization = Realization.open(directory)
    background = realization.background()
    np.testing.assert_array_equal(noise.frequencies, background.frequencies)
    assert noise.metadata.networks == background.network_names
    assert noise.metadata.observation_time == POPULATION.observation_time
    shape = (1, background.frequencies.size)
    assert noise.network("ET").pair_overlap.shape == shape
    assert noise.network("ET").pair_scale.shape == shape
    assert noise.network("ET").scale.shape == shape[1:]


def test_noise_of_a_missing_realization_fails(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["noise", str(tmp_path / "absent"), str(tmp_path / "noise.h5")]
    )

    assert result.exit_code != 0
