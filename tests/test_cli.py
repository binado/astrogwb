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
    read_population_file,
    simulate_background,
)
from astrogwb.cli import main
from astrogwb.detector import load_detector, network_pairs


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
    main(
        [
            "simulate",
            str(output),
            str(inputs["population"]),
            *("--waveform-config", str(inputs["waveform"])),
            *("--detector-config", str(inputs["detector"])),
            *("--poisson-rate", str(8.0)),
            *("--num-segments", "2"),
            *("--segment-duration", "1"),
            *("--seed", "41"),
            *("--chunk-size", "8"),
            *options,
        ],
    )


def test_a_simulate_run_writes_one_file_with_its_config(
    tmp_path: Path, inputs: dict[str, Path]
) -> None:
    output = tmp_path / "41-0.h5"

    _simulate(output, inputs)

    draw, config = BackgroundDraw.read(output)
    assert tuple(config.networks) == ("ET",)
    assert network_pairs(config.networks["ET"]) == (("R1", "S1"),)
    assert config.population == str(inputs["population"])
    assert (config.seed, config.incoherent) == (41, False)
    assert config.poisson_rate == pytest.approx(8.0)
    direct = simulate_background(
        jax.random.key(41),
        read_population_file(inputs["population"]),
        config.waveform,
        config.networks,
        poisson_rate=config.poisson_rate,
        num_segments=2,
        segment_duration=1,
        chunk_size=8,
    )
    np.testing.assert_array_equal(draw.segment_counts, direct.segment_counts)
    np.testing.assert_array_equal(draw.spectra["ET"], direct.spectra["ET"])
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
