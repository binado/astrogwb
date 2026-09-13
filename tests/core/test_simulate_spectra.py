"""Tests for the spectrum-only simulation CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astrogwb_mock_population import POPULATION_PARAMS, mock_merger_rate_fn

from astrogwb.sampling._io import load_spectra
from astrogwb.utils import years_to_seconds

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "simulate_spectra.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("simulate_spectra", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load simulate_spectra module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(*, seed: int, draws: int, observation_time: float, output: Path) -> list[str]:
    argv = [
        "--approximant",
        "TaylorF2",
        "--sampling-frequency",
        "128",
        "--minimum-frequency",
        "20",
        "--maximum-frequency",
        "48",
        "--reference-frequency",
        "20",
        "--frequency-resolution",
        "4",
        "--source-model",
        "bns_md_cosmological",
        "--rate-model",
        "madau_dickinson",
        "--model-kwarg",
        "z_min=0.3",
        "--model-kwarg",
        "z_max=20",
        "--model-kwarg",
        "n_grid=64",
        "--observation-time",
        str(observation_time),
        "--draws",
        str(draws),
        "--seed",
        str(seed),
        "--batch-size",
        "8",
        "--n-max-sigma",
        "5",
        "--output",
        str(output),
    ]
    for name, value in POPULATION_PARAMS.items():
        argv.extend(["--param", f"{name}={value}"])
    return argv


def test_import_module_without_paper_dependencies() -> None:
    module = _load_script_module()
    assert hasattr(module, "main")
    assert callable(module.main)


@pytest.mark.integration
def test_small_run_writes_expected_shapes_and_seed_changes_draws(tmp_path: Path) -> None:
    module = _load_script_module()
    rate = float(np.asarray(mock_merger_rate_fn()(POPULATION_PARAMS)))
    observation_time = 8.0 / (rate * years_to_seconds(1.0))
    draws = 3

    first_output = tmp_path / "first.h5"
    second_output = tmp_path / "second.h5"

    module.main(_argv(seed=11, draws=draws, observation_time=observation_time, output=first_output))
    module.main(_argv(seed=19, draws=draws, observation_time=observation_time, output=second_output))

    first = load_spectra(first_output)
    second = load_spectra(second_output)

    assert first.spectral_density.shape[0] == draws
    assert second.spectral_density.shape[0] == draws
    np.testing.assert_array_equal(first.frequencies, second.frequencies)
    assert first.spectral_density.shape[1] == first.frequencies.shape[0]
    assert second.spectral_density.shape[1] == second.frequencies.shape[0]
    assert not np.allclose(first.spectral_density, second.spectral_density)
