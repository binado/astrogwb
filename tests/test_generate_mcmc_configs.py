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

DEFAULT_EXAMPLE_CONFIG = _GEN.DEFAULT_EXAMPLE_CONFIG
generate_configs = _GEN.generate_configs
make_config = _GEN.make_config


def test_make_config_uses_custom_example_toml(tmp_path: Path) -> None:
    custom_example = tmp_path / "custom_mcmc.example.toml"
    custom_example.write_text(
        DEFAULT_EXAMPLE_CONFIG.read_text().replace("seed = 42", "seed = 99"),
        encoding="utf-8",
    )

    config = make_config(("E1", "E2", "E3"), ("H0",), example_config=custom_example)

    assert config.seed == 99
    assert config.catalog.detectors == ("E1", "E2", "E3")
    assert config.sampled_params == ("H0",)
    assert config.priors["H0"]["type"] == "uniform"


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
        generated.catalog.detectors
        != build_run_config(load_config(DEFAULT_EXAMPLE_CONFIG)).catalog.detectors
    )
