from __future__ import annotations

import importlib.util
from pathlib import Path

from astrogwb.sampling.config import build_run_config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

_GEN_SPEC = importlib.util.spec_from_file_location(
    "generate_mcmc_configs",
    REPO_ROOT / "scripts" / "generate_mcmc_configs.py",
)
if _GEN_SPEC is None or _GEN_SPEC.loader is None:
    raise RuntimeError("failed to load generate_mcmc_configs module spec")
_GEN = importlib.util.module_from_spec(_GEN_SPEC)
_GEN_SPEC.loader.exec_module(_GEN)

CAMPAIGNS = _GEN.CAMPAIGNS
DEFAULT_EXAMPLE_CONFIG = _GEN.DEFAULT_EXAMPLE_CONFIG
FIDUCIAL_H0 = _GEN.FIDUCIAL_H0
FIDUCIAL_LOCAL_MERGER_RATE = _GEN.FIDUCIAL_LOCAL_MERGER_RATE
RELATIVE_GAUSSIAN_SIGMA = _GEN.RELATIVE_GAUSSIAN_SIGMA
generate_configs = _GEN.generate_configs
make_config = _GEN.make_config
sweep_filenames = _GEN.sweep_filenames


def test_make_config_uses_custom_example_toml(tmp_path: Path) -> None:
    custom_example = tmp_path / "custom_mcmc.example.toml"
    custom_example.write_text(
        DEFAULT_EXAMPLE_CONFIG.read_text().replace("seed = 42", "seed = 99"),
        encoding="utf-8",
    )

    config = make_config(("E1", "E2", "E3"), ("H0",), example_config=custom_example)

    assert config.seed == 99
    assert config.analysis.detectors == ("E1", "E2", "E3")
    assert config.sampled_params == ("H0",)
    assert config.priors["H0"]["type"] == "uniform"


def test_generated_configs_are_catalog_independent() -> None:
    config = make_config(("E1", "E2", "E3"), ("H0",))

    assert config.analysis.detectors == ("E1", "E2", "E3")
    assert "catalog" not in config.model_dump(mode="json")


def test_make_config_applies_prior_overrides() -> None:
    override = {
        "local_merger_rate": {
            "type": "normal",
            "loc": FIDUCIAL_LOCAL_MERGER_RATE,
            "scale": RELATIVE_GAUSSIAN_SIGMA * FIDUCIAL_LOCAL_MERGER_RATE,
        },
    }
    config = make_config(
        ("S1", "R1"),
        ("H0", "local_merger_rate"),
        prior_overrides=override,
    )

    assert config.priors["H0"]["type"] == "uniform"
    assert config.priors["local_merger_rate"] == override["local_merger_rate"]


def test_generate_configs_propagates_custom_example_settings(tmp_path: Path) -> None:
    custom_example = tmp_path / "custom_mcmc.example.toml"
    custom_example.write_text(
        DEFAULT_EXAMPLE_CONFIG.read_text().replace(
            "num_warmup = 2000",
            "num_warmup = 123",
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "sweep"

    _, written, skipped = generate_configs(
        output_dir,
        example_config=custom_example,
        skip_existing=False,
    )

    assert skipped == []
    assert written

    generated = build_run_config(load_config(written[0]))
    assert generated.sampler.num_warmup == 123
    assert generated.sampler.num_chains == 4
    assert generated.runtime.host_device_count is None
    assert (
        generated.analysis.detectors
        != build_run_config(load_config(DEFAULT_EXAMPLE_CONFIG)).analysis.detectors
    )


def test_generate_configs_writes_campaign_subdirs(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep"
    _, written, skipped = generate_configs(output_dir, skip_existing=False)

    assert skipped == []
    assert len(written) == len(sweep_filenames())

    campaign_dirs = {path.parent.name for path in written}
    assert campaign_dirs == set(CAMPAIGNS)

    relative = {str(path.relative_to(output_dir)) for path in written}
    assert relative == set(sweep_filenames())


def test_gaussian_campaign_priors(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep"
    generate_configs(output_dir, skip_existing=False)

    merger_gauss = build_run_config(
        load_config(
            output_dir / "cosmology" / "ET-2L-aligned__H0-merger-rate-gauss.json"
        )
    )
    assert merger_gauss.sampled_params == ("H0", "local_merger_rate")
    assert merger_gauss.priors["H0"]["type"] == "uniform"
    assert merger_gauss.priors["local_merger_rate"] == {
        "type": "normal",
        "loc": FIDUCIAL_LOCAL_MERGER_RATE,
        "scale": RELATIVE_GAUSSIAN_SIGMA * FIDUCIAL_LOCAL_MERGER_RATE,
    }

    xi_h0_gauss = build_run_config(
        load_config(
            output_dir / "modified-propagation" / "ET-2L-aligned__Xi_0-H0-gauss.json"
        )
    )
    assert xi_h0_gauss.sampled_params == ("xi_0", "H0")
    assert xi_h0_gauss.priors["xi_0"]["type"] == "uniform"
    assert xi_h0_gauss.priors["H0"] == {
        "type": "normal",
        "loc": FIDUCIAL_H0,
        "scale": RELATIVE_GAUSSIAN_SIGMA * FIDUCIAL_H0,
    }


def test_campaign_sample_sets() -> None:
    assert set(CAMPAIGNS["cosmology"]) == {
        "H0",
        "H0-Omega_m",
        "H0-merger-rate",
        "H0-merger-rate-gauss",
    }
    assert set(CAMPAIGNS["modified-propagation"]) == {
        "Xi_0",
        "Xi_0-n",
        "Xi_0-H0-gauss",
    }
    assert set(CAMPAIGNS["astrophysical"]) == {"H0-peak", "H0-MD", "Xi_0-MD"}
