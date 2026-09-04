"""Runtime catalog preparation: loading and analysis-window truncation.

Composition used to live here: a run declared an inline spec and the two
catalogs were mixed in memory from persisted banks. Every catalog is a file
now, built by ``scripts/generate_catalog.py``, so loading is just reading one
and the draw law is pinned in ``test_catalog_generation.py`` instead. What
remains here is what still happens at run time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from catalog_fixtures import make_catalog, save_catalog

from astrogwb.paper.catalogs import load_run_catalog, truncate_catalog_samples


# --------------------------------------------------------------------------- #
# truncate_catalog_samples
# --------------------------------------------------------------------------- #
def _catalog(redshift: np.ndarray, *, offset: float = 0.0) -> xr.Dataset:
    return make_catalog(
        frequencies=np.linspace(10.0, 50.0, 5),
        polarization_power=np.arange(5 * redshift.size, dtype=float).reshape(
            5, redshift.size
        )
        + offset,
        source_parameters={
            "redshift": redshift,
            "luminosity_distance": 1.0e3 * (1.0 + redshift),
        },
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=50.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
        df=10.0,
    )


def test_truncate_catalog_samples_keeps_only_window_samples() -> None:
    catalog = _catalog(np.array([0.05, 0.25, 0.4, 1.5, 21.0]))

    truncated = truncate_catalog_samples(
        catalog, label="proposal", minimum_redshift=0.3, maximum_redshift=20.0
    )

    kept = truncated.source_parameters.sel(parameter="redshift").values
    np.testing.assert_allclose(kept, [0.4, 1.5])
    # Rows stay consistent across every variable sharing the sample dim.
    assert truncated.polarization_power.shape == (5, 2)
    np.testing.assert_allclose(
        truncated.polarization_power.values,
        catalog.polarization_power.isel(sample=[2, 3]).values,
    )


def test_truncate_catalog_samples_rejects_empty_window() -> None:
    catalog = _catalog(np.array([0.05, 0.25]))

    with pytest.raises(ValueError, match="no samples in the analysis redshift window"):
        truncate_catalog_samples(
            catalog, label="injection", minimum_redshift=0.3, maximum_redshift=20.0
        )


def test_truncate_catalog_samples_requires_distance_column() -> None:
    redshift = np.array([0.4, 1.5])
    catalog = make_catalog(
        frequencies=np.linspace(10.0, 50.0, 5),
        polarization_power=np.ones((5, redshift.size)),
        source_parameters={"redshift": redshift},
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=50.0,
        reference_frequency=20.0,
        sampling_frequency=128.0,
        df=10.0,
    )

    with pytest.raises(ValueError, match="missing required parameter"):
        truncate_catalog_samples(
            catalog, label="injection", minimum_redshift=0.3, maximum_redshift=20.0
        )


# --------------------------------------------------------------------------- #
# load_run_catalog
# --------------------------------------------------------------------------- #
def test_load_run_catalog_returns_the_whole_file(tmp_path: Path) -> None:
    """No prefixing: a catalog file is exactly the catalog a run samples."""
    redshift = np.arange(6, dtype=float)
    path = tmp_path / "catalog.h5"
    save_catalog(path, _catalog(redshift))

    loaded = load_run_catalog(path, label="proposal")

    assert loaded.sizes["sample"] == 6
    np.testing.assert_array_equal(
        loaded.source_parameters.sel(parameter="redshift").values, redshift
    )


def test_load_run_catalog_names_the_role_on_a_missing_file(tmp_path: Path) -> None:
    """Fails before JAX claims a device, so the message must say which role."""
    with pytest.raises(FileNotFoundError, match="proposal catalog not found"):
        load_run_catalog(tmp_path / "absent.h5", label="proposal")


def test_load_run_catalog_is_eager(tmp_path: Path) -> None:
    """Loaded, not lazily opened: the file handle must not outlive the call."""
    path = tmp_path / "catalog.h5"
    save_catalog(path, _catalog(np.arange(4, dtype=float)))

    loaded = load_run_catalog(path, label="injection")
    path.unlink()

    assert float(loaded.polarization_power.values.sum()) >= 0.0
