from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

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
