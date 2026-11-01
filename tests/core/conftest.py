"""Shared pytest configuration for the core package.

Enables JAX double precision for the whole suite, as
``astrogwb.paper.runtime`` does: polarization power underflows float32. The
mock-population fixtures wrap :mod:`astrogwb_mock_population`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest

# Module scope, above the helper import: must precede any array creation.
jax.config.update("jax_enable_x64", True)

from astrogwb_mock_population import (
    F_MAX,
    F_MIN,
    build_mock_catalog,
    load_mock_population,
)

from astrogwb.gwb.catalog import (
    CatalogMetadata,
    PolarizationPowerData,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def frequencies() -> np.ndarray:
    return np.linspace(F_MIN, F_MAX, 128)


@pytest.fixture
def load_orf_fixture() -> Callable[[str], dict[str, np.ndarray]]:
    """Load a committed ORF fixture by name; a missing file is a test failure."""

    def _loader(name: str) -> dict[str, np.ndarray]:
        return dict(np.load(FIXTURES_DIR / f"{name}.npz"))

    return _loader


@pytest.fixture(scope="session")
def mock_population() -> dict[str, np.ndarray]:
    """Draw the mock population once per pytest worker."""
    return load_mock_population()


@pytest.fixture(scope="session")
def mock_catalog_factory(
    mock_population: dict[str, np.ndarray],
) -> Callable[..., tuple[PolarizationPowerData, CatalogMetadata]]:
    """Build a ``(data, metadata)`` pair from the mock population draw, once per worker."""

    # `partial` keeps build_mock_catalog's signature for type checkers.
    return partial(build_mock_catalog, mock_population)
