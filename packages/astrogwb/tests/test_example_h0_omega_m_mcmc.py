"""Smoke test for the amplitude-marginalized (H0, Omega_m) example.

Like its sibling, the example lives outside the installed package and is loaded
by path. The failure mode specific to this one is the marginalization
bookkeeping: ``H0`` is a ``numpyro.factor``, not a chain site, so a run that
skipped the reconstruction pass would write chains that are structurally valid
and simply have no ``H0`` in them.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import jax
import numpy as np
import pytest
import xarray as xr
from astrogwb.waveform import make_catalog, save_catalog

# See the note in `test_example_h0_mcmc.py`: the `toy_catalog` fixture's
# distances must be float64 to match what the example recomputes.
jax.config.update("jax_enable_x64", True)

EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "h0_omega_m_mcmc.py"

#: Prior bounds the reconstructed H0 must fall inside, matching the example's
#: `--h0-min` / `--h0-max` defaults.
H0_MIN = 20.0
H0_MAX = 140.0


def load_example() -> ModuleType:
    """Import the example by path; it is not an installed module."""
    spec = importlib.util.spec_from_file_location(
        "h0_omega_m_mcmc_example", EXAMPLE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXAMPLE = load_example()


def test_example_path_exists() -> None:
    assert EXAMPLE_PATH.is_file()


def test_rejects_catalog_without_required_parameters(tmp_path: Path) -> None:
    path = tmp_path / "bare.h5"
    frequencies = np.linspace(10.0, 100.0, 4)
    save_catalog(
        path,
        make_catalog(
            frequencies=frequencies,
            polarization_power=np.ones((4, 3)),
            source_parameters={"redshift": np.linspace(0.1, 1.0, 3)},
            approximant="ToyApproximant",
            minimum_frequency=10.0,
            maximum_frequency=100.0,
            reference_frequency=20.0,
            sampling_frequency=4096.0,
        ),
    )
    with pytest.raises(ValueError, match="luminosity_distance"):
        EXAMPLE.load_samples(path, None)


def test_rejects_single_detector(
    tmp_path: Path, toy_catalog: Callable[..., None]
) -> None:
    catalog_path = tmp_path / "catalog.h5"
    toy_catalog(catalog_path, EXAMPLE.FIDUCIALS)
    with pytest.raises(ValueError, match="at least"):
        EXAMPLE.main(
            [
                str(catalog_path),
                "-o",
                str(tmp_path / "chains.nc"),
                "--detectors",
                "E1",
            ]
        )


@pytest.mark.integration
def test_runs_end_to_end_and_writes_chains(
    tmp_path: Path, toy_catalog: Callable[..., None]
) -> None:
    catalog_path = tmp_path / "catalog.h5"
    output_path = tmp_path / "chains" / "h0_omega_m.nc"
    toy_catalog(catalog_path, EXAMPLE.FIDUCIALS)

    EXAMPLE.main(
        [
            str(catalog_path),
            "-o",
            str(output_path),
            "--detectors",
            "E1",
            "E2",
            "E3",
            "--num-warmup",
            "30",
            "--num-samples",
            "20",
            "--n-grid",
            "32",
            "--no-progress-bar",
        ]
    )

    assert output_path.is_file()
    chains = xr.open_dataset(output_path, engine="h5netcdf")
    try:
        # H0 is not a chain site; its presence here is the reconstruction pass
        # having run and merged back into the posterior.
        assert set(chains.data_vars) >= {
            "H0",
            "Omega_m",
            "total_merger_rate",
            "importance_relative_ess",
            "quadrature_effective_nodes",
        }
        assert chains["H0"].shape == (1, 20)
        assert chains["Omega_m"].shape == (1, 20)

        h0 = chains["H0"].values
        assert np.all(np.isfinite(h0))
        # Drawn from AmplitudeConditional, whose support is the prior's. The
        # value is not asserted tightly: with Omega_m free and 20 draws on a toy
        # catalog, a point check would be flaky.
        assert np.all((h0 > H0_MIN) & (h0 < H0_MAX))

        # The catalog is its own proposal, so every weight is 1 by construction
        # at the fiducial and stays close to it across a short chain.
        assert float(chains["importance_relative_ess"].min()) > 0.5
        assert np.all(np.isfinite(chains["quadrature_effective_nodes"].values))

        assert chains.attrs["amplitude_parameter"] == "H0"
        assert chains.attrs["fiducial_H0"] == pytest.approx(EXAMPLE.FIDUCIALS["H0"])
        assert chains.attrs["detectors"] == "E1 E2 E3"
    finally:
        chains.close()
