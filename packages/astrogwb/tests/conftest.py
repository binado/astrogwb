"""Shared pytest configuration for the core package.

Enables JAX double precision for the whole suite: numerical assertions are
written for x64 (e.g. polarization power of order 1e-47 Hz^-2 underflows
float32's smallest normal), and the ``require_x64``-guarded entry points are
only meaningful there. Same setup as ``astrogwb_paper.runtime``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
import pytest

# Must precede any array creation, hence module scope rather than a fixture:
# pytest imports conftest before collecting test modules.
jax.config.update("jax_enable_x64", True)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def frequencies() -> np.ndarray:
    return np.geomspace(20, 2048, 128)


@pytest.fixture
def load_orf_fixture() -> Callable[[str], dict[str, np.ndarray]]:
    """Load a committed ORF fixture by name; a missing file is a test failure."""

    def _loader(name: str) -> dict[str, np.ndarray]:
        return dict(np.load(FIXTURES_DIR / f"{name}.npz"))

    return _loader
