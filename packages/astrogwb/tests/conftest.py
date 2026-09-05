"""Shared pytest configuration for the core package.

Enables JAX double precision for the whole suite: numerical assertions are
written for x64 (e.g. polarization power of order 1e-47 Hz^-2 underflows
float32's smallest normal), and the ``require_x64``-guarded entry points are
only meaningful there. Same setup as ``astrogwb_paper.runtime``.

The mock-population fixtures below are thin wrappers over
:mod:`astrogwb_mock_population`. Test modules should import shared constants
and builders from that module rather than from ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest

# Must precede any array creation, hence module scope rather than a fixture:
# pytest imports conftest before collecting test modules. It is kept above the
# `astrogwb_mock_population` import because that module reaches into astrogwb and JAX;
# neither builds an array at import time, but the ordering is what makes that
# irrelevant rather than something to re-check on every edit.
jax.config.update("jax_enable_x64", True)

from astrogwb.catalog import Catalog
from astrogwb_mock_population import (
    F_MAX,
    F_MIN,
    build_mock_catalog,
    build_synthetic_weights_callback,
    load_mock_population,
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
    """Load the committed mock population once per pytest worker."""
    return load_mock_population()


@pytest.fixture(scope="session")
def mock_catalog_factory(
    mock_population: dict[str, np.ndarray],
) -> Callable[..., Catalog]:
    """Build a real ``Catalog`` from the committed population draw.

    Session-scoped, so the draw is parsed once. The root ``addopts`` is
    ``-n auto --dist loadscope``, so this is once *per worker* rather than once
    globally -- the catalog build is milliseconds, and ``loadscope`` keeps a
    module on one worker, so the NUTS runs that actually cost are unaffected.
    """

    # `partial` rather than a **kwargs passthrough, so the returned factory
    # keeps build_mock_catalog's real keyword signature for type checkers.
    return partial(build_mock_catalog, mock_population)


@pytest.fixture
def synthetic_weights_callback() -> Callable[..., tuple[object, dict[str, jax.Array]]]:
    """Expose the synthetic weights callback builder as a fixture."""
    return build_synthetic_weights_callback
