"""CLI runtime initialization."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest
import tomli_w
from astrogwb_tiny_pipeline import WAVEFORM, write_population_csv

from astrogwb.background.simulation import (
    BackgroundDraw,
    simulate_background,
)
from astrogwb.cli import main
from astrogwb.detector import load_detector, network_pairs
from astrogwb.populations.io import read_population_file
from astrogwb.waveform import waveform_generator


def test_importing_the_cli_leaves_the_xla_backend_uninitialized() -> None:
    """A late ``set_host_device_count(2)`` still taking effect proves it."""
    command = "simulate"
    code = f"""
from astrogwb.cli import main

try:
    main(["{command}", "--help"])
except SystemExit as error:
    assert error.code == 0
import numpyro

numpyro.set_host_device_count(2)
import jax

assert jax.device_count() == 2, f"backend initialized early: {{jax.devices()}}"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_cli_rejects_invalid_segment_count(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "simulate",
                "out.h5",
                "pool.csv",
                "--waveform-config",
                "waveform.toml",
                "--detector-config",
                "detectors.toml",
                "--network",
                "ET",
                "--poisson-rate",
                "1",
                "--num-segments",
                "0",
                "--segment-duration",
                "1",
                "--seed",
                "1",
            ]
        )
    assert error.value.code == 2
    assert "--num-segments" in capsys.readouterr().err


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    """A gwmock-pop catalogue, waveform settings and two network definitions."""
    waveform = tmp_path / "waveform.toml"
    waveform.write_bytes(
        tomli_w.dumps(WAVEFORM.model_dump(mode="json", exclude_none=True)).encode()
    )
    detector = tmp_path / "detector.toml"
    names = ["S1", "R1", "C1"]
    detectors = {
        name: load_detector(name).model_dump(mode="json", exclude_none=True)
        for name in names
    }
    detector.write_bytes(
        tomli_w.dumps(
            {"detectors": detectors, "networks": {"ET": names[:2], "triangle": names}}
        ).encode()
    )
    return {
        "population": write_population_csv(tmp_path / "events.csv"),
        "waveform": waveform,
        "detector": detector,
    }


def _simulate(
    output: Path, inputs: dict[str, Path], *options: str, network: str = "ET"
) -> None:
    main(
        [
            "simulate",
            str(output),
            str(inputs["population"]),
            *("--waveform-config", str(inputs["waveform"])),
            *("--detector-config", str(inputs["detector"])),
            *("--network", network),
            *("--poisson-rate", str(8.0)),
            *("--num-segments", "2"),
            *("--segment-duration", "1"),
            *("--seed", "41"),
            *("--chunk-size", "8"),
            *options,
        ],
    )


@pytest.mark.parametrize("noise", [False, True])
def test_a_simulate_run_writes_one_file_with_its_config(
    tmp_path: Path, inputs: dict[str, Path], noise: bool
) -> None:
    output = tmp_path / "41-0.h5"

    flags = ["--simulate-detector-noise"] if noise else []
    _simulate(output, inputs, *flags)

    draw, config = BackgroundDraw.read(output)
    assert config.network_name == "ET"
    assert network_pairs(config.network) == (("R1", "S1"),)
    assert config.population == str(inputs["population"])
    assert config.seed == 41
    assert config.simulate_detector_noise is noise
    assert config.poisson_rate == pytest.approx(8.0)
    direct = simulate_background(
        jax.random.key(41),
        read_population_file(inputs["population"]),
        waveform_generator(config.waveform),
        config.network,
        poisson_rate=config.poisson_rate,
        num_segments=2,
        segment_duration=1,
        chunk_size=8,
        simulate_detector_noise=noise,
    )
    np.testing.assert_array_equal(draw.segment_counts, direct.segment_counts)
    np.testing.assert_array_equal(
        draw.pair_spectral_density, direct.pair_spectral_density
    )
    assert draw.segment_counts.shape == (2,)
    assert draw.segment_counts.sum() > 0
    assert np.any(draw.pair_spectral_density)
    with h5py.File(output, "r") as file:
        metadata = json.loads(file.attrs["metadata"])
    assert set(metadata["network"]) == {"R1", "S1"}
    assert [path.name for path in tmp_path.glob("*.h5")] == ["41-0.h5"]


def test_a_simulate_run_rejects_an_unknown_network(
    tmp_path: Path, inputs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "41-0.h5"

    with pytest.raises(SystemExit) as error:
        _simulate(output, inputs, network="unknown")

    assert error.value.code == 2
    assert "unknown network 'unknown'" in capsys.readouterr().err
    assert not output.exists()


def test_simulate_reads_a_custom_population_root(
    tmp_path: Path, inputs: dict[str, Path]
) -> None:
    catalogue = np.genfromtxt(inputs["population"], delimiter=",", names=True)
    inputs["population"] = tmp_path / "events.h5"
    with h5py.File(inputs["population"], "w") as file:
        file.create_dataset("catalogue/events", data=catalogue)

    output = tmp_path / "result.h5"
    _simulate(output, inputs, "--population-root", "catalogue/events")

    assert BackgroundDraw.read(output)[0].segment_counts.sum() > 0
