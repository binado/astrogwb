"""Runtime catalog loading and the checks between the two files a run names.

Composition used to live here: a run declared an inline spec and the two
catalogs were mixed in memory from persisted banks. Every catalog is a file
now, built by ``scripts/generate_catalog.py``, so loading is just reading one.
Narrowing to the analysis window moved onto
:func:`~astrogwb.simulators.polarization_power.restrict_redshift`, which moves
the samples and the recorded density together and is covered in ``tests/core``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from catalog_fixtures import make_catalog, save_catalog

from astrogwb.paper.catalogs import load_run_catalog, validate_matching_frequency_grids


def test_load_run_catalog_returns_the_whole_file(tmp_path: Path) -> None:
    """No prefixing: a catalog file is exactly the catalog a run samples."""
    redshift = np.linspace(0.1, 5.0, 6)
    path = tmp_path / "catalog.h5"
    save_catalog(make_catalog(redshift=redshift), path)

    data, metadata = load_run_catalog(path, label="proposal")

    assert metadata.num_samples == 6
    np.testing.assert_allclose(data["source_parameters"]["redshift"], redshift)


def test_load_run_catalog_names_the_role_on_a_missing_file(tmp_path: Path) -> None:
    """Fails before JAX claims a device, so the message must say which role."""
    with pytest.raises(FileNotFoundError, match="proposal catalog not found"):
        load_run_catalog(tmp_path / "absent.h5", label="proposal")


def test_load_run_catalog_names_the_role_on_a_mismatched_request(
    tmp_path: Path,
) -> None:
    """A file built from another draw than the run asks for is refused."""
    path = tmp_path / "catalog.h5"
    _, metadata = catalog = make_catalog(redshift=np.linspace(0.1, 5.0, 4))
    save_catalog(catalog, path, seed=41)

    load_run_catalog(path, label="injection", request=(metadata, np.uint64(41)))
    with pytest.raises(ValueError, match="injection catalog.*seed"):
        load_run_catalog(path, label="injection", request=(metadata, np.uint64(42)))


def test_load_run_catalog_is_eager(tmp_path: Path) -> None:
    """Loaded, not lazily opened: the file handle must not outlive the call."""
    path = tmp_path / "catalog.h5"
    save_catalog(make_catalog(redshift=np.linspace(0.1, 5.0, 4)), path)

    data, _ = load_run_catalog(path, label="injection")
    path.unlink()

    assert float(data["polarization_power"].sum()) >= 0.0


def test_injection_and_proposal_frequency_grids_must_match() -> None:
    with pytest.raises(ValueError, match="identical frequency grids"):
        validate_matching_frequency_grids(jnp.array([2.0, 3.0]), jnp.array([2.0, 4.0]))
