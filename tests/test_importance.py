"""Tests for the importance-weighting reference model and cosmology helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from gwmock_pop.distributions.madau_dickinson import madau_dickinson_rate

from astrogwb.cosmology import distance_and_volume_grid, log_gw_em_ratio
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    compute_merger_rate_and_log_density,
    make_merger_rate_and_log_weights_fn,
)
from astrogwb.utils import SECONDS_PER_YEAR

# Standard cosmology + population hyperparameters used across the tests.
FIDUCIALS = {
    "H0": 67.66,
    "Omega_m": 0.3096,
    "xi_0": 1.0,
    "xi_n": 1.91,
    "gamma": 1.42,
    "kappa": 4.62,
    "z_peak": 1.84,
    "local_merger_rate": 161.0,
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
    d_l, dvc_dz = distance_and_volume_grid(FIDUCIALS, max_redshift=Z_MAX, n_grid=N_GRID)
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
    d_l, _ = distance_and_volume_grid(
        FIDUCIALS,
        max_redshift=jnp.asarray(Z_MAX),  # ty: ignore[invalid-argument-type]
        n_grid=N_GRID,
    )
    assert np.asarray(d_l).shape == (N_GRID,)


# --------------------------------------------------------------------------- #
# make_merger_rate_and_log_weights_fn
# --------------------------------------------------------------------------- #
def _build_synthetic_callback(n_samples: int = 16):
    z_grid = jnp.linspace(Z_MIN, Z_MAX, N_GRID)
    z_samples = jnp.linspace(0.01, Z_MAX - 0.01, n_samples)
    samples = {"redshift": z_samples}

    _, proposal_logprob = compute_merger_rate_and_log_density(
        FIDUCIALS, samples, z_grid=z_grid
    )
    fn = make_merger_rate_and_log_weights_fn(
        z_grid=z_grid,
        proposal_logprob=proposal_logprob,
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


def test_local_merger_rate_scales_total_rate_without_changing_weights() -> None:
    fn, samples = _build_synthetic_callback()
    fiducial_rate, fiducial_log_weights = fn(FIDUCIALS, samples)

    scaled_params = {**FIDUCIALS, "local_merger_rate": 2.5 * 161.0}
    scaled_rate, scaled_log_weights = fn(scaled_params, samples)

    assert float(scaled_rate) == pytest.approx(2.5 * float(fiducial_rate))
    np.testing.assert_allclose(scaled_log_weights, fiducial_log_weights)


def test_fiducial_local_merger_rate_preserves_rate_calculation() -> None:
    fn, samples = _build_synthetic_callback()
    total_rate, _ = fn(FIDUCIALS, samples)

    z_grid = jnp.linspace(Z_MIN, Z_MAX, N_GRID)
    _, dvc_dz_grid = distance_and_volume_grid(
        FIDUCIALS, max_redshift=Z_MAX, n_grid=N_GRID
    )
    rate_shape_grid = madau_dickinson_rate(
        z_grid,
        FIDUCIALS["gamma"],
        FIDUCIALS["kappa"],
        FIDUCIALS["z_peak"],
    )
    integral_mpc3 = jnp.trapezoid(
        rate_shape_grid / (1.0 + z_grid) * dvc_dz_grid,
        z_grid,
    )
    expected = 1e-9 * 161.0 * float(integral_mpc3) / SECONDS_PER_YEAR

    assert float(total_rate) == pytest.approx(expected)


def test_make_merger_rate_and_log_weights_fn_fiducial_weights_cancel() -> None:
    # Proposal and target share compute_merger_rate_and_log_density, so at the
    # fiducial point log_weights are identically zero and relative ESS is 1.
    fn, samples = _build_synthetic_callback()
    _, log_weights = fn(FIDUCIALS, samples)
    log_weights = np.asarray(log_weights)
    weights = np.exp(log_weights)
    np.testing.assert_allclose(log_weights, 0.0, atol=1e-12)
    assert np.all(np.isfinite(weights))
    rel_ess = float(weights.sum() ** 2 / (weights.size * (weights**2).sum()))
    assert rel_ess == pytest.approx(1.0)
