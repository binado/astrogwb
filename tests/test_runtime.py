from __future__ import annotations

from dataclasses import dataclass

import pytest

from astrogwb.runtime import configure_runtime


@dataclass
class _FakeDevice:
    platform: str


@pytest.mark.parametrize(
    ("device_platform", "expected_chain_method"),
    [("tpu", "vectorized"), ("gpu", "vectorized"), ("cpu", "parallel")],
)
def test_configure_runtime_auto_chain_method(
    monkeypatch: pytest.MonkeyPatch,
    device_platform: str,
    expected_chain_method: str,
) -> None:
    """chain_method="auto" resolves from the reported device platform."""
    import jax

    monkeypatch.setattr(jax, "devices", lambda: [_FakeDevice(device_platform)])

    _, chain_method = configure_runtime(num_chains=1, platform="auto")

    assert chain_method == expected_chain_method


def test_configure_runtime_explicit_chain_method_not_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit chain_method wins even on an accelerator platform."""
    import jax

    monkeypatch.setattr(jax, "devices", lambda: [_FakeDevice("tpu")])

    _, chain_method = configure_runtime(
        num_chains=1, platform="auto", chain_method="sequential"
    )

    assert chain_method == "sequential"
