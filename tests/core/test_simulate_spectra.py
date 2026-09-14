"""Tests for the spectrum-only simulation CLI."""

from __future__ import annotations

import importlib.util
import json
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
    return [
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
        "--model-kwargs",
        json.dumps({"z_min": 0.3, "z_max": 20, "n_grid": 64}),
        "--params",
        json.dumps(POPULATION_PARAMS),
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


def test_import_module_without_paper_dependencies() -> None:
    module = _load_script_module()
    assert hasattr(module, "main")
    assert callable(module.main)
    source = SCRIPT_PATH.read_text()
    assert "astrogwb.paper" not in source


def test_parse_args_loads_json_dicts() -> None:
    module = _load_script_module()
    args = module.parse_args(
        [
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
            "--model-kwargs",
            '{"z_min": 0.3, "n_grid": 64}',
            "--params",
            '{"log10_R0": 1.5, "gamma": 2.7}',
            "--observation-time",
            "1.0",
            "--draws",
            "2",
            "--seed",
            "0",
            "--output",
            "/tmp/spectra.h5",
        ]
    )
    assert args.model_kwargs == {"z_min": 0.3, "n_grid": 64}
    assert args.params == {"log10_R0": 1.5, "gamma": 2.7}
    assert args.average_mode == "analytic_inclination"


def test_padded_event_capacity_is_the_poisson_tail() -> None:
    module = _load_script_module()
    assert module.padded_event_capacity(8.0, 5.0) == int(
        np.ceil(8.0 + 5.0 * np.sqrt(8.0))
    )
    assert module.padded_event_capacity(0.0, 5.0) == 1


@pytest.mark.integration
def test_small_run_writes_expected_shapes_and_seed_changes_draws(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    rate = float(np.asarray(mock_merger_rate_fn()(POPULATION_PARAMS)))
    observation_time = 8.0 / (rate * years_to_seconds(1.0))
    draws = 3

    first_output = tmp_path / "first.h5"
    second_output = tmp_path / "second.h5"

    module.main(
        _argv(
            seed=11, draws=draws, observation_time=observation_time, output=first_output
        )
    )
    module.main(
        _argv(
            seed=19,
            draws=draws,
            observation_time=observation_time,
            output=second_output,
        )
    )

    first = load_spectra(first_output)
    second = load_spectra(second_output)

    assert first.spectral_density.shape[0] == draws
    assert second.spectral_density.shape[0] == draws
    np.testing.assert_array_equal(first.frequencies, second.frequencies)
    assert first.spectral_density.shape[1] == first.frequencies.shape[0]
    assert second.spectral_density.shape[1] == second.frequencies.shape[0]
    assert first.n_events.shape == (draws,)
    assert first.average_mode == "analytic_inclination"
    assert first.observation_time == observation_time
    # Strain spectra are ~1e-50; use a relative scale so distinct seeds fail the check.
    scale = max(
        np.max(np.abs(first.spectral_density)),
        np.max(np.abs(second.spectral_density)),
        1e-300,
    )
    assert not np.allclose(
        first.spectral_density / scale,
        second.spectral_density / scale,
        rtol=1e-6,
        atol=1e-6,
    )
