"""Smoke test for the standalone H0 MCMC example.

The example lives in ``examples/``, which is outside the installed package, so
it is loaded by path. The failure modes it guards are all silent ones: an
empty analysis band, a catalog missing ``redshift``/``luminosity_distance``,
and importance weights that collapse to ``-inf``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from astrogwb.cosmology import distance_and_volume_grid
from astrogwb.waveform import make_catalog, save_catalog

# The fixture builds luminosity distances the example later recomputes in
# float64; a float32 fixture would disagree just enough to bias the weights.
# Safe here because importing jax creates no arrays -- only the first array
# commits the backend.
jax.config.update("jax_enable_x64", True)

EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "h0_mcmc.py"


def load_example() -> ModuleType:
    """Import ``examples/h0_mcmc.py`` by path; it is not an installed module."""
    spec = importlib.util.spec_from_file_location("h0_mcmc_example", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXAMPLE = load_example()

#: Sets the toy signal's total optimal SNR to roughly 20 against the ET
#: triangle over a 1 yr observation. A louder catalog makes the noiseless
#: likelihood so sharp that NUTS cannot adapt a step size within a short
#: warmup and every draw registers as a divergence.
TOY_POWER_SCALE = 1e-48


def write_toy_catalog(path: Path, *, n_frequency: int = 24, n_sample: int = 16) -> None:
    """A minimal but structurally valid ``waveform_catalog`` file.

    ``luminosity_distance`` must be the flat-LambdaCDM distance at the
    example's own fiducials, not an arbitrary function of redshift. The
    importance weights carry a ``-2 (log d_L(z|theta) - log d_L,catalog)``
    term, so a catalog whose distances disagree with the cosmology yields
    weights far from 1 even at the fiducial -- a collapsed effective sample
    size that looks like a bug in the example rather than in the fixture.
    """
    frequencies = np.linspace(10.0, 2048.0, n_frequency)
    redshift = np.linspace(0.1, 3.0, n_sample)

    fiducials = EXAMPLE.FIDUCIALS
    grid = jnp.asarray(np.linspace(redshift[0], redshift[-1], 64))
    distance_grid, _ = distance_and_volume_grid(fiducials, grid)
    luminosity_distance = np.asarray(
        jnp.interp(jnp.asarray(redshift), grid, distance_grid)
    )

    # Falling power spectrum, weakest for the most distant sources.
    power = (frequencies[:, None] / frequencies[0]) ** -7.0 / (1.0 + redshift[None, :])
    catalog = make_catalog(
        frequencies=frequencies,
        polarization_power=TOY_POWER_SCALE * power,
        source_parameters={
            "redshift": redshift,
            "luminosity_distance": luminosity_distance,
        },
        approximant="ToyApproximant",
        minimum_frequency=float(frequencies[0]),
        maximum_frequency=float(frequencies[-1]),
        reference_frequency=20.0,
        sampling_frequency=4096.0,
    )
    save_catalog(path, catalog)


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


def test_rejects_single_detector(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.h5"
    write_toy_catalog(catalog_path)
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
def test_runs_end_to_end_and_writes_chains(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.h5"
    output_path = tmp_path / "chains" / "h0.nc"
    write_toy_catalog(catalog_path)

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
        assert chains["H0"].shape == (1, 20)
        assert set(chains.data_vars) >= {
            "H0",
            "total_merger_rate",
            "importance_relative_ess",
        }
        assert np.all(np.isfinite(chains["H0"].values))
        # The catalog is its own proposal, so every weight is 1 by construction
        # at the fiducial and stays close to it across a short chain.
        assert float(chains["importance_relative_ess"].min()) > 0.5
        assert chains.attrs["fiducial_H0"] == pytest.approx(EXAMPLE.FIDUCIALS["H0"])
        assert chains.attrs["detectors"] == "E1 E2 E3"
    finally:
        chains.close()
