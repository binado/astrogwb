from __future__ import annotations

from dataclasses import dataclass

import pytest

from astrogwb.paper.runtime import (
    CPU_THREAD_ENV_VARS,
    colab_tpu_available,
    configure_runtime,
)


@dataclass
class _FakeDevice:
    platform: str


@pytest.mark.parametrize(
    ("device_platform", "device_count", "num_chains", "expected_chain_method"),
    [
        ("tpu", 4, 4, "parallel"),
        ("gpu", 2, 2, "parallel"),
        ("tpu", 1, 4, "vectorized"),
        ("gpu", 1, 4, "vectorized"),
        ("cpu", 4, 4, "parallel"),
        ("cpu", 1, 4, "sequential"),
    ],
)
def test_configure_runtime_auto_chain_method(
    monkeypatch: pytest.MonkeyPatch,
    device_platform: str,
    device_count: int,
    num_chains: int,
    expected_chain_method: str,
) -> None:
    """chain_method="auto" resolves from platform and visible device count."""
    import jax

    devices = [_FakeDevice(device_platform) for _ in range(device_count)]
    monkeypatch.setattr(jax, "devices", lambda: devices)

    _, chain_method = configure_runtime(num_chains=num_chains, platform="auto")

    assert chain_method == expected_chain_method


@pytest.mark.parametrize(
    "explicit_chain_method", ["parallel", "sequential", "vectorized"]
)
def test_configure_runtime_explicit_chain_method_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
    explicit_chain_method: str,
) -> None:
    """An explicit chain_method wins even on an accelerator platform."""
    import jax

    monkeypatch.setattr(jax, "devices", lambda: [_FakeDevice("tpu")])

    _, chain_method = configure_runtime(
        num_chains=4, platform="auto", chain_method=explicit_chain_method
    )

    assert chain_method == explicit_chain_method


def test_configure_runtime_cpu_threads_pins_all_thread_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--cpu-threads must own every BLAS/vector thread env var.

    Snakemake injects OMP/GOTO/OPENBLAS/MKL/VECLIB/NUMEXPR thread counts equal
    to the job's ``threads`` into every job environment. A partial override
    leaves those pools at the job thread count, oversubscribing the host when
    several single-threaded chains run concurrently.
    """
    import jax

    for var in CPU_THREAD_ENV_VARS:
        monkeypatch.setenv(var, "4")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(jax, "devices", lambda: [_FakeDevice("cpu")])

    configure_runtime(num_chains=1, platform="cpu", cpu_threads=1)

    import os

    for var in CPU_THREAD_ENV_VARS:
        assert os.environ[var] == "1"
    assert "intra_op_parallelism_threads=1" in os.environ["XLA_FLAGS"]


def test_colab_tpu_available_from_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COLAB_TPU_ADDR", raising=False)
    monkeypatch.setattr(
        "astrogwb.paper.runtime.os.path.exists", lambda path: path == "/dev/accel0"
    )

    assert colab_tpu_available()


def test_colab_tpu_available_from_legacy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("astrogwb.paper.runtime.os.path.exists", lambda path: False)
    monkeypatch.setenv("COLAB_TPU_ADDR", "10.0.0.2:8470")

    assert colab_tpu_available()


def test_colab_tpu_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("astrogwb.paper.runtime.os.path.exists", lambda path: False)
    monkeypatch.delenv("COLAB_TPU_ADDR", raising=False)

    assert not colab_tpu_available()
