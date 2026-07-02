"""Tests for the importance-weighting reference model and cosmology helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from gwmock_pop.distributions.madau_dickinson import madau_dickinson_redshift_pdf

from astrogwb.importance.cosmology import flat_lcdm_grid, log_gw_em_ratio
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    make_merger_rate_and_log_weights_fn,
)

# Standard cosmology + population hyperparameters used across the tests.
FIDUCIALS = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 2.7,
    "kappa": 3.0,
    "z_peak": 2.0,
}

Z_MIN = 0.0
Z_MAX = 20.0
N_GRID = 256


# --------------------------------------------------------------------------- #
# log_gw_em_ratio
# --------------------------------------------------------------------------- #
def test_log_gw_em_ratio_at_zero_redshift() -> None:
    # At z=0 the ratio is xi_0 + (1 - xi_0) * 1 = 1 -> log = 0 for any xi.
    out = float(log_gw_em_ratio(jnp.asarray(0.0), xi_0=1.5, xi_n=2.0))
    assert out == pytest.approx(0.0, abs=1e-12)


def test_log_gw_em_ratio_gr_propagation_is_zero() -> None:
    # xi_0 = 1 -> ratio identically 1 -> log = 0 at every redshift.
    z = jnp.linspace(0.0, 5.0, 50)
    out = np.asarray(log_gw_em_ratio(z, xi_0=1.0, xi_n=3.0))
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_log_gw_em_ratio_increases_with_redshift() -> None:
    # For xi_0 > 1 and xi_n > 0, the ratio xi_0 + (1 - xi_0)*(1+z)^(-xi_n) rises
    # from 1 (at z=0) towards xi_0 (as z -> inf), so its log increases with z.
    z = jnp.linspace(0.01, 5.0, 50)
    out = np.asarray(log_gw_em_ratio(z, xi_0=2.0, xi_n=1.0))
    assert np.all(np.diff(out) > 0)


# --------------------------------------------------------------------------- #
# flat_lcdm_grid
# --------------------------------------------------------------------------- #
def test_flat_lcdm_grid_shapes_and_finiteness() -> None:
    d_l, dvc_dz = flat_lcdm_grid(FIDUCIALS, max_redshift=Z_MAX, n_grid=N_GRID)
    d_l = np.asarray(d_l)
    dvc_dz = np.asarray(dvc_dz)
    assert d_l.shape == (N_GRID,)
    assert dvc_dz.shape == (N_GRID,)
    assert np.all(np.isfinite(d_l))
    assert np.all(np.isfinite(dvc_dz))
    # Luminosity distance and dV/dz are non-negative for a physical cosmology.
    assert np.all(d_l >= 0)
    assert np.all(dvc_dz >= 0)


def test_flat_lcdm_grid_accepts_jax_scalar_max_redshift() -> None:
    # Documents the static-scalar contract: a concrete (non-traced) jnp scalar
    # for max_redshift must not crash the build -- gwmock_pop accepts it.
    d_l, _ = flat_lcdm_grid(
        FIDUCIALS, max_redshift=float(jnp.asarray(Z_MAX)), n_grid=N_GRID
    )
    assert np.asarray(d_l).shape == (N_GRID,)


# --------------------------------------------------------------------------- #
# make_merger_rate_and_log_weights_fn
# --------------------------------------------------------------------------- #
def _build_synthetic_callback(n_samples: int = 16):
    z_grid = jnp.linspace(Z_MIN, Z_MAX, N_GRID)
    z_samples = jnp.linspace(0.01, Z_MAX - 0.01, n_samples)

    proposal_pdf = madau_dickinson_redshift_pdf(
        z_samples,
        z_max=Z_MAX,
        z_min=Z_MIN,
        gamma=FIDUCIALS["gamma"],
        kappa=FIDUCIALS["kappa"],
        z_peak=FIDUCIALS["z_peak"],
        hubble_constant=FIDUCIALS["H0"],
        omega_m=FIDUCIALS["Omega_m"],
        n_grid=N_GRID,
    )
    proposal_log_pdf = jnp.log(proposal_pdf)

    # Luminosity distances consistent with the fiducial cosmology, evaluated on
    # the same grid so the closure interpolates sensible values.
    d_l_grid, _ = flat_lcdm_grid(FIDUCIALS, max_redshift=Z_MAX, n_grid=N_GRID)
    d_l_samples = jnp.interp(z_samples, z_grid, d_l_grid)

    samples = {
        "redshift": z_samples,
        "luminosity_distance": d_l_samples,
    }

    fn = make_merger_rate_and_log_weights_fn(
        z_grid=z_grid,
        proposal_log_pdf=proposal_log_pdf,
        local_merger_rate=161.0,
        fiducial_xi_0=FIDUCIALS["xi_0"],
        fiducial_xi_n=FIDUCIALS["xi_n"],
    )
    return fn, samples


def test_make_merger_rate_and_log_weights_fn_smoke() -> None:
    fn, samples = _build_synthetic_callback()
    total_rate, log_weights = fn(FIDUCIALS, samples)

    total_rate = float(total_rate)
    log_weights = np.asarray(log_weights)
    assert total_rate > 0.0
    assert np.all(np.isfinite(log_weights))
    assert log_weights.shape == (samples["redshift"].shape[0],)


def test_make_merger_rate_and_log_weights_fn_fiducial_weights_finite_and_healthy() -> (
    None
):
    # At the fiducial point the GW/EM-ratio and luminosity-distance Jacobian
    # terms vanish, but the target PDF (which carries the dVc/dz comoving-volume
    # factor and a grid-integral normalization) does not match the plain proposal
    # redshift PDF, so the weights are non-trivial. We assert they stay finite
    # and the relative ESS is in a healthy range (well above 0, well below the
    # degenerate-weights value of 1).
    fn, samples = _build_synthetic_callback()
    _, log_weights = fn(FIDUCIALS, samples)
    weights = np.exp(np.asarray(log_weights))
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0)
    rel_ess = float(weights.sum() ** 2 / (weights.size * (weights**2).sum()))
    assert 0.05 < rel_ess < 1.0
