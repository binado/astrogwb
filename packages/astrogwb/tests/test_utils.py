"""Tests for shared utility helpers."""

from __future__ import annotations

import jax
import pytest
from astrogwb.utils import require_x64


def test_require_x64_raises_when_disabled() -> None:
    @require_x64
    def identity(value: float) -> float:
        return value

    enabled = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="x64"):
            identity(1.0)
    finally:
        jax.config.update("jax_enable_x64", enabled)
