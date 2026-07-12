from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrogwb.config.loading import load_mapping
from astrogwb.config.mcmc import build_run_config, config_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_mcmc import (  # noqa: E402
    ResolvedRuntime,
    RuntimeOptions,
    build_run_record,
    configure_runtime,
    parse_args,
    runtime_options_from_args,
)


@pytest.mark.parametrize("flag", ["--host-device-count", "--cpu-threads"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_runtime_cli_rejects_non_positive_counts(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--config", "config.toml", "--catalog", "catalog.h5", flag, value])


def test_runtime_cli_defaults_and_choices() -> None:
    args = parse_args(["--config", "config.toml", "--catalog", "catalog.h5"])
    options = runtime_options_from_args(args)

    assert options == RuntimeOptions(
        platform="auto",
        host_device_count=None,
        cpu_threads=None,
        chain_method="auto",
    )

    args = parse_args(
        [
            "--config",
            "config.toml",
            "--catalog",
            "catalog.h5",
            "--platform",
            "gpu",
            "--host-device-count",
            "2",
            "--cpu-threads",
            "5",
            "--chain-method",
            "sequential",
        ]
    )
    assert runtime_options_from_args(args) == RuntimeOptions(
        platform="gpu",
        host_device_count=2,
        cpu_threads=5,
        chain_method="sequential",
    )


class _FakeJaxConfig:
    def __init__(self) -> None:
        self.x64 = False

    def update(self, name: str, value: bool) -> None:
        assert name == "jax_enable_x64"
        self.x64 = value

    def read(self, name: str) -> bool:
        assert name == "jax_enable_x64"
        return self.x64


def _install_fake_runtime_modules(monkeypatch: pytest.MonkeyPatch, platform: str):
    host_counts: list[int] = []
    fake_numpyro = SimpleNamespace(set_host_device_count=host_counts.append)
    fake_jax = SimpleNamespace(
        config=_FakeJaxConfig(),
        devices=lambda: [SimpleNamespace(platform=platform)],
    )
    monkeypatch.setitem(sys.modules, "numpyro", fake_numpyro)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    return fake_jax, host_counts


def test_configure_runtime_replaces_xla_thread_limit_and_preserves_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "XLA_FLAGS",
        "--xla_force_host_platform_device_count=8 "
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=99 --xla_dump_to=/tmp/xla",
    )
    fake_jax, host_counts = _install_fake_runtime_modules(monkeypatch, "cpu")

    jax, resolved = configure_runtime(
        RuntimeOptions(platform="cpu", cpu_threads=4), num_chains=2
    )

    assert jax is fake_jax
    assert host_counts == [2]
    assert resolved == ResolvedRuntime("cpu", 2, 4, "parallel")
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
    assert "intra_op_parallelism_threads=99" not in os.environ["XLA_FLAGS"]
    assert "intra_op_parallelism_threads=4" in os.environ["XLA_FLAGS"]
    assert "--xla_cpu_multi_thread_eigen=false" not in os.environ["XLA_FLAGS"]
    assert os.environ["XLA_FLAGS"].count("--xla_cpu_multi_thread_eigen=true") == 1
    assert "--xla_force_host_platform_device_count=8" in os.environ["XLA_FLAGS"]
    assert "--xla_dump_to=/tmp/xla" in os.environ["XLA_FLAGS"]


def test_configure_runtime_omitted_thread_cap_preserves_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited = {
        "OMP_NUM_THREADS": "7",
        "OPENBLAS_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "5",
        "XLA_FLAGS": "--xla_dump_to=/tmp/xla intra_op_parallelism_threads=9",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    _install_fake_runtime_modules(monkeypatch, "gpu")

    _, resolved = configure_runtime(RuntimeOptions(), num_chains=3)

    assert resolved == ResolvedRuntime("gpu", 3, None, "vectorized")
    for name, value in inherited.items():
        assert os.environ[name] == value
    assert "JAX_PLATFORMS" not in os.environ


def test_gpu_cli_selection_maps_to_cuda_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_runtime_modules(monkeypatch, "gpu")

    configure_runtime(RuntimeOptions(platform="gpu"), num_chains=1)

    assert os.environ["JAX_PLATFORMS"] == "cuda"


def test_sidecar_separates_requested_and_resolved_runtime_from_config_hash() -> None:
    config = build_run_config(load_mapping(REPO_ROOT / "configs/mcmc.example.toml"))
    requested = RuntimeOptions("auto", None, None, "auto")
    resolved = ResolvedRuntime("cpu", 4, None, "parallel")

    record = build_run_record(
        config,
        catalog_path=Path("catalog.h5"),
        timestamp="20260712-120000",
        catalog_sha256="catalog-hash",
        runtime_options=requested,
        resolved_runtime=resolved,
    )

    assert record["runtime"] == {
        "requested": {
            "platform": "auto",
            "host_device_count": None,
            "cpu_threads": None,
            "chain_method": "auto",
        },
        "resolved": {
            "platform": "cpu",
            "host_device_count": 4,
            "cpu_threads": None,
            "chain_method": "parallel",
        },
    }
    assert record["config_sha256"] == config_sha256(config)
    assert "runtime" not in config.model_dump(mode="json")
