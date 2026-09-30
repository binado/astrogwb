"""Ensemble SNR conventions, typed study config, and checked cache reuse."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle
from repo import REPO_ROOT

from astrogwb.catalog import SpectralDensityCatalog, SpectrumGenerator
from astrogwb.gwb import spectral_snr
from astrogwb.metadata import SpectraMetadata, artifact_path
from astrogwb.paper import plotting, snr
from astrogwb.paper.config.detectors import DetectorRegistry
from astrogwb.paper.config.runs import base_config_paths, merge_config_layers
from astrogwb.paper.plotting.snr import plot_snr_histogram
from astrogwb.utils import years_to_seconds

REFERENCE_NETWORK = "ET-2L-aligned-CE-Hanford"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "analyze_spectrum_snrs.py"
    spec = importlib.util.spec_from_file_location("analyze_spectrum_snrs_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw() -> dict[str, Any]:
    return merge_config_layers(
        [
            *base_config_paths(REPO_ROOT),
            REPO_ROOT / "config/simulations/spectrum/fixed.toml",
        ]
    )


@pytest.fixture
def catalog(raw: dict[str, Any]) -> SpectralDensityCatalog:
    metadata = SpectraMetadata.model_validate({**raw["spectra"], "num_draws": 3})
    return SpectralDensityCatalog(
        spectral_density=np.array(
            [
                [1.0, 8.0, 1.0, 2.0],
                [4.0, 1.0, 2.0, 1.0],
                [2.0, 2.0, 5.0, 3.0],
            ]
        ),
        frequencies=np.array([10.0, 13.0, 20.0, 32.0]),
        n_events=np.full(3, metadata.num_events),
        total_merger_rate=np.full(3, 1.0),
        hyperparameters={
            name: np.full(3, value) for name, value in metadata.fixed.items()
        },
        _metadata=metadata,
    )


@pytest.fixture
def registry(raw: dict[str, Any]) -> DetectorRegistry:
    return DetectorRegistry.model_validate(
        {"detectors": raw["detectors"], "networks": raw["networks"]}
    )


@pytest.mark.parametrize("bounds", [(10.0, 32.0), (13.0, 20.0), (19.0, 21.0)])
def test_ensemble_matches_single_spectra_on_full_nonuniform_grid(
    catalog: SpectralDensityCatalog,
    registry: DetectorRegistry,
    monkeypatch: pytest.MonkeyPatch,
    bounds: tuple[float, float],
) -> None:
    calls = []
    noise = np.array([2.0, 1.0, 3.0, 4.0])

    def effective_noise(frequencies, geometry, sensitivities):
        calls.append((np.asarray(frequencies), geometry, sensitivities))
        return noise

    monkeypatch.setattr(snr, "effective_psd", effective_noise)
    values, mean_snr = snr.compute_spectrum_snrs(
        catalog,
        registry,
        REFERENCE_NETWORK,
        minimum_frequency=bounds[0],
        maximum_frequency=bounds[1],
    )
    mask = jnp.asarray(
        (catalog.frequencies >= bounds[0]) & (catalog.frequencies <= bounds[1])
    )
    seconds = years_to_seconds(catalog.observation_time)
    expected = [
        float(
            spectral_snr(
                jnp.asarray(row),
                jnp.asarray(noise),
                seconds,
                catalog.frequencies,
                frequency_mask=mask,
            )
        )
        for row in catalog.spectral_density
    ]
    np.testing.assert_allclose(values, expected, rtol=1e-13)
    assert mean_snr == pytest.approx(
        float(
            spectral_snr(
                jnp.asarray(catalog.spectral_density.mean(axis=0)),
                jnp.asarray(noise),
                seconds,
                catalog.frequencies,
                frequency_mask=mask,
            )
        )
    )
    if bounds != (19.0, 21.0):
        assert mean_snr < values.mean()
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], catalog.frequencies)
    assert (
        tuple(detector.name for detector in calls[0][1])
        == registry.networks[REFERENCE_NETWORK]
    )


def test_summary_uses_unbiased_sd_and_separate_mean_spectrum_statistic() -> None:
    result = snr.summarize_spectrum_snrs([1.0, 2.0, 6.0], mean_spectrum_snr=2.5)
    assert result == pytest.approx(
        {
            "num_draws": 3,
            "mean": 3.0,
            "median": 2.0,
            "sd": np.sqrt(7.0),
            "q05": 1.1,
            "q95": 5.6,
            "relative_scatter": np.sqrt(7.0) / 3.0,
            "mean_spectrum_snr": 2.5,
        }
    )


@pytest.mark.parametrize("values", [[0.0, 0.0], [1.0], [0.0]])
def test_undefined_scatter_is_missing(values: list[float]) -> None:
    result = snr.summarize_spectrum_snrs(values, mean_spectrum_snr=values[0])
    assert np.isnan(result["relative_scatter"])
    if len(values) == 1:
        assert np.isnan(result["sd"])
    else:
        assert result["sd"] == 0.0


@pytest.mark.parametrize(
    "values", [[], [[1.0]], [float("nan")], [float("inf")], [-1.0]]
)
def test_invalid_summary_inputs_are_rejected(values: list[Any]) -> None:
    with pytest.raises(ValueError):
        snr.summarize_spectrum_snrs(values, mean_spectrum_snr=1.0)


@pytest.mark.parametrize("bounds", [(40.0, 50.0), (20.0, 10.0), (float("nan"), 20.0)])
def test_invalid_or_empty_bands_are_rejected(
    catalog: SpectralDensityCatalog,
    registry: DetectorRegistry,
    bounds: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="frequency"):
        snr.compute_spectrum_snrs(
            catalog,
            registry,
            REFERENCE_NETWORK,
            minimum_frequency=bounds[0],
            maximum_frequency=bounds[1],
        )


def test_nonfinite_snrs_are_rejected(
    catalog: SpectralDensityCatalog,
    registry: DetectorRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(snr, "effective_psd", lambda *_: np.zeros(4))
    with pytest.raises(ValueError, match="nonfinite"):
        snr.compute_spectrum_snrs(
            catalog,
            registry,
            REFERENCE_NETWORK,
            minimum_frequency=10.0,
            maximum_frequency=32.0,
        )


def test_config_extracts_typed_inputs_and_preserves_detector_overrides(
    script: ModuleType,
    raw: dict[str, Any],
    tmp_path: Path,
) -> None:
    psd = tmp_path / "custom.txt"
    np.savetxt(psd, [[1.0, 1e-45], [2048.0, 2e-45]])
    raw["detectors"]["S1"]["psd_reference"] = str(psd)
    raw["detectors"]["S1"]["geometry"]["xarm_azimuth_rad"] = 0.5
    raw["networks"]["custom"] = ["S1", "R1"]
    config = script.SNRConfig.from_merged(raw, network="custom")
    assert isinstance(config.spectra, SpectraMetadata)
    assert isinstance(config.detector_registry, DetectorRegistry)
    assert config.detector_registry.networks["custom"] == ("S1", "R1")
    assert config.detector_registry.detectors["S1"].psd_reference == str(psd)
    assert config.detector_registry.detectors["S1"].geometry.xarm_azimuth_rad == 0.5
    dumped = config.model_dump(mode="json")
    assert set(dumped) == {
        "spectra",
        "detector_registry",
        "network",
        "minimum_frequency",
        "maximum_frequency",
    }
    assert script.SNRConfig.model_validate_json(config.model_dump_json()) == config
    with pytest.raises(ValueError, match="extra_forbidden"):
        script.SNRConfig.model_validate({**dumped, "typo": 1})
    with pytest.raises(ValueError, match="frozen"):
        config.network = "another"


@pytest.mark.parametrize(
    "change, match",
    [
        ({"minimum_frequency": 2048.0}, "less than"),
        ({"maximum_frequency": float("inf")}, "finite"),
        ({"network": "missing"}, "known networks"),
    ],
)
def test_config_rejects_invalid_analysis(
    script: ModuleType,
    raw: dict[str, Any],
    change: dict[str, Any],
    match: str,
) -> None:
    config = script.SNRConfig.from_merged(raw, network=REFERENCE_NETWORK)
    with pytest.raises(ValueError, match=match):
        script.SNRConfig.model_validate({**config.model_dump(), **change})


def test_config_rejects_missing_spectra_and_sampled_hyperparameters(
    script: ModuleType,
    raw: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=r"no \[spectra\]"):
        script.SNRConfig.from_merged(
            {k: v for k, v in raw.items() if k != "spectra"}, network=REFERENCE_NETWORK
        )
    raw["spectra"]["hyperparameters"]["local_merger_rate"] = {
        "dist": "Uniform",
        "kwargs": {"low": 500.0, "high": 1000.0},
    }
    with pytest.raises(ValueError, match="sampled parameters: local_merger_rate"):
        script.SNRConfig.from_merged(raw, network=REFERENCE_NETWORK)


@pytest.fixture
def run_script(
    script: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Path]:
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    monkeypatch.chdir(REPO_ROOT)
    layer = tmp_path / "case.toml"
    layer.write_text("[spectra]\nnum_draws = 3\n", encoding="utf-8")
    layers = [
        *base_config_paths(REPO_ROOT),
        REPO_ROOT / "config/simulations/spectrum/fixed.toml",
        layer,
    ]
    args = [arg for path in layers for arg in ("--config", str(path))]
    monkeypatch.setattr(script, "configure_runtime", lambda **_: None)
    original_style = plotting.use_paper_style

    def headless_style() -> None:
        original_style()
        plt.rcParams["text.usetex"] = False

    monkeypatch.setattr(plotting, "use_paper_style", headless_style)

    def run(*extra: str, output_dir: Path | None = None) -> Path:
        output = output_dir or tmp_path / "analysis"
        with plt.rc_context():
            script.main(
                [
                    *args,
                    "--spectra-dir",
                    str(tmp_path / "spectra"),
                    "--output-dir",
                    str(output),
                    *extra,
                ]
            )
        return output

    return run


def test_generation_then_other_network_reuses_cached_spectra(
    script: ModuleType,
    catalog: SpectralDensityCatalog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_script: Callable[..., Path],
) -> None:
    events = []
    monkeypatch.setattr(
        script, "configure_runtime", lambda **_: events.append("runtime")
    )

    def generate(self, metadata):
        assert events == ["runtime"]
        assert metadata == catalog.metadata
        events.append("generate")
        return catalog

    monkeypatch.setattr(SpectrumGenerator, "__call__", generate)
    output = run_script()
    artifact = artifact_path(catalog.metadata, tmp_path / "spectra")
    original_bytes = artifact.read_bytes()
    draws = pd.read_csv(output / "snr_draws.csv")
    summary = pd.read_csv(output / "snr_summary.csv")
    provenance = json.loads((output / "provenance.json").read_text())
    assert list(draws.draw_index) == [0, 1, 2]
    assert set(draws.spectrum_key) == {catalog.metadata.key()}
    assert set(draws.network) == {REFERENCE_NETWORK}
    assert set(draws.n_events) == {catalog.metadata.num_events}
    assert summary.loc[0, "mean"] == pytest.approx(draws.snr.mean())
    assert summary.loc[0, "sd"] == pytest.approx(draws.snr.std(ddof=1))
    assert provenance["config"]["spectra"] == catalog.metadata.model_dump(mode="json")
    assert provenance["config"]["network"] == REFERENCE_NETWORK
    assert provenance["source_path"] == str(artifact)
    assert provenance["distribution"] == "Finite-catalog estimator scatter"
    assert not provenance["detector_noise_realizations"]
    assert set(provenance["selected_detectors"]) == {"S1", "R1", "C1"}
    assert (output / "snr_histogram.pdf").stat().st_size > 0

    other = run_script(
        "--network", "ET-triangular", "--cache-only", output_dir=tmp_path / "other"
    )
    assert events == ["runtime", "generate", "runtime"]
    assert artifact.read_bytes() == original_bytes
    other_provenance = json.loads((other / "provenance.json").read_text())
    assert other_provenance["spectrum_key"] == provenance["spectrum_key"]
    assert other_provenance["config"]["network"] == "ET-triangular"


@pytest.mark.parametrize("flag", ["--cache-only", "--cached-only"])
def test_cache_only_miss_fails_before_runtime_or_generation(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    run_script: Callable[..., Path],
    flag: str,
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("runtime and generation must not run on a cache-only miss")

    monkeypatch.setattr(script, "configure_runtime", forbidden)
    monkeypatch.setattr(SpectrumGenerator, "__call__", forbidden)
    with pytest.raises(FileNotFoundError, match="generation is disabled"):
        run_script(flag)


def test_metadata_mismatch_does_not_trigger_regeneration(
    script: ModuleType,
    catalog: SpectralDensityCatalog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_script: Callable[..., Path],
) -> None:
    altered = copy.deepcopy(catalog.metadata.model_dump())
    altered["population"]["seed"] += 1
    wrong = replace(catalog, _metadata=SpectraMetadata.model_validate(altered))
    path = artifact_path(catalog.metadata, tmp_path / "spectra")
    path.parent.mkdir()
    wrong.save(path)

    def forbidden(*args, **kwargs):
        pytest.fail("metadata mismatch must fail before runtime or generation")

    monkeypatch.setattr(script, "configure_runtime", forbidden)
    monkeypatch.setattr(SpectrumGenerator, "__call__", forbidden)
    with pytest.raises(ValueError, match="not the requested"):
        run_script()


def test_histogram_has_one_raw_snr_panel_and_preserves_counts() -> None:
    import matplotlib.pyplot as plt

    figure = plot_snr_histogram(
        [1.0, 2.0, 2.0, 5.0],
        network="test",
        distribution_label="Finite-observation realizations",
    )
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_xlabel() == "SNR"
        assert axis.get_ylabel() == "Realizations"
        assert (
            sum(
                patch.get_height()
                for patch in axis.patches
                if isinstance(patch, Rectangle)
            )
            == 4
        )
        assert "Finite-observation" in axis.get_title()
    finally:
        plt.close(figure)


def test_script_import_and_config_validation_leave_backend_free() -> None:
    code = """
import runpy
from pathlib import Path
from astrogwb.paper.config.runs import base_config_paths, merge_config_layers
module = runpy.run_path('scripts/analyze_spectrum_snrs.py')
raw = merge_config_layers([*base_config_paths(), Path('config/simulations/spectrum/fixed.toml')])
module['SNRConfig'].from_merged(raw, network=module['DEFAULT_NETWORK'])
import numpyro
numpyro.set_host_device_count(2)
import jax
assert jax.device_count() == 2, 'config validation initialized the backend'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
