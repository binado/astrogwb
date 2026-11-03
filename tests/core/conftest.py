"""Shared pytest configuration for the core package.

Enables JAX double precision for the whole suite, as
``astrogwb.runtime`` does: intensity underflows float32. The
mock-population fixtures wrap :mod:`astrogwb_mock_population`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

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

from astrogwb.importance.catalog import (
    CatalogMetadata,
    IntensityData,
)


@pytest.fixture
def frequencies() -> np.ndarray:
    return np.linspace(F_MIN, F_MAX, 128)


@pytest.fixture(scope="session")
def mock_population() -> dict[str, np.ndarray]:
    """Draw the mock population once per pytest worker."""
    return load_mock_population()


@pytest.fixture(scope="session")
def mock_catalog_factory(
    mock_population: dict[str, np.ndarray],
) -> Callable[..., tuple[IntensityData, CatalogMetadata]]:
    """Build a ``(data, metadata)`` pair from the mock population draw, once per worker."""

    # `partial` keeps build_mock_catalog's signature for type checkers.
    return partial(build_mock_catalog, mock_population)
