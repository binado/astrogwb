from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.cosmology import distance_and_volume_grid
from astrogwb.waveform import make_catalog, save_catalog

FIXTURES_DIR = Path(__file__).parent / "fixtures"

#: Sets the toy signal's total optimal SNR to roughly 20 against the ET
#: triangle over a 1 yr observation. A louder catalog makes the noiseless
#: likelihood so sharp that NUTS cannot adapt a step size within a short
#: warmup and every draw registers as a divergence.
TOY_POWER_SCALE = 1e-48


@pytest.fixture
def frequencies() -> np.ndarray:
    return np.geomspace(20, 2048, 128)


@pytest.fixture
def load_orf_fixture() -> Callable[[str], dict[str, np.ndarray]]:
    """Load a committed ORF fixture by name; a missing file is a test failure."""

    def _loader(name: str) -> dict[str, np.ndarray]:
        return dict(np.load(FIXTURES_DIR / f"{name}.npz"))

    return _loader


@pytest.fixture
def toy_catalog() -> Callable[..., None]:
    """Write a minimal but structurally valid ``waveform_catalog`` file.

    Shared by the ``examples/`` smoke tests, which run against the same
    ``FIDUCIALS``. Callers must have enabled ``jax_enable_x64`` before the
    fixture runs: the distances built here are the ones the example later
    recomputes in float64, and a float32 fixture would disagree just enough to
    bias the weights.

    ``luminosity_distance`` must be the flat-LambdaCDM distance at the caller's
    own fiducials, not an arbitrary function of redshift -- hence the explicit
    ``fiducials`` argument. The importance weights carry a
    ``-2 (log d_L(z|theta) - log d_L,catalog)`` term, so a catalog whose
    distances disagree with the cosmology yields weights far from 1 even at the
    fiducial -- a collapsed effective sample size that looks like a bug in the
    example rather than in the fixture.
    """

    def _write(
        path: Path,
        fiducials: Mapping[str, float],
        *,
        n_frequency: int = 24,
        n_sample: int = 16,
    ) -> None:
        frequencies = np.linspace(10.0, 2048.0, n_frequency)
        redshift = np.linspace(0.1, 3.0, n_sample)

        grid = jnp.asarray(np.linspace(redshift[0], redshift[-1], 64))
        distance_grid, _ = distance_and_volume_grid(fiducials, grid)
        luminosity_distance = np.asarray(
            jnp.interp(jnp.asarray(redshift), grid, distance_grid)
        )

        # Falling power spectrum, weakest for the most distant sources.
        power = (frequencies[:, None] / frequencies[0]) ** -7.0 / (
            1.0 + redshift[None, :]
        )
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

    return _write
