"""Shared pytest configuration for the library suite.

Puts ``tests/`` on ``sys.path``: pytest's ``prepend`` import mode adds only
each test directory, so ``tests/parity`` cannot otherwise import a helper such
as ``reference_population``.

Enables JAX double precision for the whole suite, as
``astrogwb.runtime`` does: intensity underflows float32. The
mock-population fixtures wrap :mod:`astrogwb_mock_population`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Module scope, above the helper import: must precede any array creation.
jax.config.update("jax_enable_x64", True)

from astrogwb_mock_population import (
    F_MAX,
    F_MIN,
    build_mock_catalog,
    load_mock_population,
)

from astrogwb.importance.catalog import IntensityData


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
) -> Callable[..., IntensityData]:
    """Build an intensity catalog from the mock population draw, once per worker."""

    # `partial` keeps build_mock_catalog's signature for type checkers.
    return partial(build_mock_catalog, mock_population)
