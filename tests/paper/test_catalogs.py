"""Runtime catalog loading and the checks between the two files a run names.

Composition used to live here: a run declared an inline spec and the two
catalogs were mixed in memory from persisted banks. Every catalog is a file
now, built by ``scripts/generate_catalog.py``, so loading is just reading one.
Narrowing to the analysis window moved onto
:meth:`~astrogwb.catalog.PolarizationPowerCatalog.restrict_redshift`, which
moves the samples
and the recorded density together and is covered in ``tests/core``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest
from catalog_fixtures import make_catalog

from astrogwb.paper.catalogs import load_run_catalog, validate_matching_frequency_grids


def test_load_run_catalog_returns_the_whole_file(tmp_path: Path) -> None:
    """No prefixing: a catalog file is exactly the catalog a run samples."""
    redshift = np.linspace(0.1, 5.0, 6)
    path = tmp_path / "catalog.h5"
    make_catalog(redshift=redshift).save(path)

    loaded = load_run_catalog(path, label="proposal")

    assert loaded.num_samples == 6
    np.testing.assert_allclose(loaded.source_parameters["redshift"], redshift)


def test_load_run_catalog_names_the_role_on_a_missing_file(tmp_path: Path) -> None:
    """Fails before JAX claims a device, so the message must say which role."""
    with pytest.raises(FileNotFoundError, match="proposal catalog not found"):
        load_run_catalog(tmp_path / "absent.h5", label="proposal")


def test_load_run_catalog_names_the_role_on_a_stale_file(tmp_path: Path) -> None:
    """A file that cannot say what drew it fails here, before JAX claims a device."""
    path = tmp_path / "catalog.h5"
    make_catalog(redshift=np.linspace(0.1, 5.0, 4)).save(path)
    with h5py.File(path, "r+") as handle:
        handle.attrs["format_name"] = "astrogwb_catalog_v3"

    with pytest.raises(ValueError, match="injection catalog.*format_name"):
        load_run_catalog(path, label="injection")


def test_load_run_catalog_is_eager(tmp_path: Path) -> None:
    """Loaded, not lazily opened: the file handle must not outlive the call."""
    path = tmp_path / "catalog.h5"
    make_catalog(redshift=np.linspace(0.1, 5.0, 4)).save(path)

    loaded = load_run_catalog(path, label="injection")
    path.unlink()

    assert float(loaded.polarization_power.sum()) >= 0.0


def test_injection_and_proposal_frequency_grids_must_match() -> None:
    with pytest.raises(ValueError, match="identical frequency grids"):
        validate_matching_frequency_grids(jnp.array([2.0, 3.0]), jnp.array([2.0, 4.0]))
