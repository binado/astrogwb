"""Tests for the importance-weighting reference model and cosmology helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Standard cosmology + population hyperparameters, and the redshift grid they
# are integrated on. Shared with `synthetic_importance`, which builds its
# catalog at exactly these values: a second copy here would let the two drift
# apart with no visible symptom.
from astrogwb_mock_population import (
    FIDUCIALS,
    Z_MAX,
    log_weight_kwargs,
)

from astrogwb.cosmology import distance_and_volume_grid, log_gw_em_ratio
from astrogwb.distributions.rates import madau_dickinson_rate
from astrogwb.importance.spectral import (
    evaluate_log_weights,
    importance_spectral_density,
)


# --------------------------------------------------------------------------- #
# madau_dickinson_rate
# --------------------------------------------------------------------------- #
def test_madau_dickinson_rate_scales_with_local_merger_rate() -> None:
    z = jnp.asarray([0.0, 1.0, 3.0])
    base = madau_dickinson_rate(z, 1.42, 4.62, 1.84)
    scaled = madau_dickinson_rate(z, 1.42, 4.62, 1.84, 3.5)
    np.testing.assert_allclose(scaled, 3.5 * base)


def test_madau_dickinson_rate_is_unity_at_redshift_zero() -> None:
    z = jnp.asarray(0.0)
    assert float(madau_dickinson_rate(z, 1.42, 4.62, 1.84)) == pytest.approx(1.0)
    assert float(madau_dickinson_rate(z, 2.7, 2.9, 1.9)) == pytest.approx(1.0)


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


def test_flat_lcdm_grid_evaluates_on_passed_grid() -> None:
    # D4 regression: the function must evaluate on the caller's grid, so
    # rate_shape_grid and dvc_dz_grid can never land on two different grids.
    z_grid = jnp.array([0.0, 0.3, 1.0, 2.7, 8.0, Z_MAX])
    d_l, dvc_dz = distance_and_volume_grid(
        z_grid,
        hubble_constant=FIDUCIALS["H0"],
        omega_m=FIDUCIALS["Omega_m"],
    )
    d_l = np.asarray(d_l)
    dvc_dz = np.asarray(dvc_dz)
    assert d_l.shape == z_grid.shape
    assert dvc_dz.shape == z_grid.shape
    assert np.all(np.isfinite(d_l))
    assert np.all(np.isfinite(dvc_dz))
    # Luminosity distance grows monotonically on an ascending grid.
    assert np.all(np.diff(d_l) > 0)
    assert np.all(d_l >= 0)
    assert np.all(dvc_dz >= 0)


# --------------------------------------------------------------------------- #
# The BNS population reweighting a fixed catalog
#
# `synthetic_importance` builds a catalog that is its own proposal at FIDUCIALS,
# so these exercise the rate and the weights against a reference whose neutral
# point is known exactly.
# --------------------------------------------------------------------------- #
def _reweight(
    importance: dict[str, Any], params: dict[str, float]
) -> tuple[jax.Array, jax.Array]:
    """The total rate and log-weights at ``params``."""
    _, extras = importance_spectral_density(params, **importance)
    log_weights = evaluate_log_weights(params, **log_weight_kwargs(importance))
    return extras["total_merger_rate"], log_weights


def test_local_merger_rate_scales_total_rate_without_changing_weights(
    synthetic_importance,
) -> None:
    """Why `local_merger_rate` is analytically marginalizable: it is pure amplitude."""
    importance, _ = synthetic_importance()
    fiducial_rate, fiducial_log_weights = _reweight(importance, FIDUCIALS)

    scaled_rate, scaled_log_weights = _reweight(
        importance,
        {**FIDUCIALS, "local_merger_rate": 2.5 * FIDUCIALS["local_merger_rate"]},
    )

    assert float(scaled_rate) == pytest.approx(2.5 * float(fiducial_rate))
    # The source table now carries the absolute rate, so renormalizing a
    # differently scaled table introduces only floating-point roundoff.
    np.testing.assert_allclose(
        scaled_log_weights, fiducial_log_weights, rtol=0.0, atol=1e-14
    )
