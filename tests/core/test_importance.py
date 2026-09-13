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
    N_GRID,
    Z_MAX,
    Z_MIN,
    log_weight_kwargs,
)
from reference_population import reference_merger_rate_distance_and_logprob

from astrogwb.constants import SECONDS_PER_YEAR
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
def test_flat_lcdm_grid_shapes_and_finiteness() -> None:
    z_grid = jnp.linspace(Z_MIN, Z_MAX, N_GRID)
    d_l, dvc_dz = distance_and_volume_grid(
        z_grid,
        hubble_constant=FIDUCIALS["H0"],
        omega_m=FIDUCIALS["Omega_m"],
    )
    d_l = np.asarray(d_l)
    dvc_dz = np.asarray(dvc_dz)
    assert d_l.shape == (N_GRID,)
    assert dvc_dz.shape == (N_GRID,)
    assert np.all(np.isfinite(d_l))
    assert np.all(np.isfinite(dvc_dz))
    # Luminosity distance and dV/dz are non-negative for a physical cosmology.
    assert np.all(d_l >= 0)
    assert np.all(dvc_dz >= 0)


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


def test_flat_lcdm_grid_is_jit_traceable() -> None:
    # The function must run under jax.jit with traced params and a concrete
    # grid: no static Python scalars (max_redshift / n_grid) are extracted.
    z_grid = jnp.linspace(Z_MIN, Z_MAX, N_GRID)
    d_l, dvc_dz = jax.jit(distance_and_volume_grid)(
        z_grid,
        hubble_constant=FIDUCIALS["H0"],
        omega_m=FIDUCIALS["Omega_m"],
    )
    assert np.asarray(d_l).shape == (N_GRID,)
    assert np.asarray(dvc_dz).shape == (N_GRID,)


# --------------------------------------------------------------------------- #
# The reference redshift density
# --------------------------------------------------------------------------- #
def test_redshift_logpdf_normalizes_on_its_own_grid() -> None:
    # Interpolating the unnormalized density and dividing by its trapezoidal
    # integral makes the interpolant integrate to exactly that normalization.
    z_grid = jnp.linspace(Z_MIN, Z_MAX, N_GRID)
    _, _, logpdf = reference_merger_rate_distance_and_logprob(
        FIDUCIALS, z_grid, redshift_grid=z_grid
    )

    assert np.trapezoid(np.exp(np.asarray(logpdf)), np.asarray(z_grid)) == (
        pytest.approx(1.0)
    )


def test_redshift_logpdf_is_negative_infinite_outside_the_grid() -> None:
    # Samples off the grid interpolate to zero density; callers rely on the
    # -inf (rather than a clamped edge value) to zero out such rows.
    z_grid = jnp.linspace(Z_MIN, Z_MAX, N_GRID)
    outside = jnp.asarray([Z_MAX + 0.5, Z_MIN - 0.5])

    _, _, logpdf = reference_merger_rate_distance_and_logprob(
        FIDUCIALS, outside, redshift_grid=z_grid
    )

    assert np.all(np.isneginf(np.asarray(logpdf)))


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


def test_reweighting_a_synthetic_catalog_is_finite(
    synthetic_importance,
) -> None:
    importance, samples = synthetic_importance()
    total_rate, log_weights = _reweight(importance, FIDUCIALS)

    total_rate = float(total_rate)
    log_weights = np.asarray(log_weights)
    assert total_rate > 0.0
    assert np.all(np.isfinite(log_weights))
    assert log_weights.shape == (samples["redshift"].shape[0],)


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


def test_fiducial_local_merger_rate_preserves_rate_calculation(
    synthetic_importance,
) -> None:
    """The merger-rate function matches the hand-written grid formula."""
    importance, _ = synthetic_importance()
    total_rate, _ = _reweight(importance, FIDUCIALS)

    z_grid = jnp.linspace(Z_MIN, Z_MAX, N_GRID)
    _, dvc_dz_grid = distance_and_volume_grid(
        z_grid,
        hubble_constant=FIDUCIALS["H0"],
        omega_m=FIDUCIALS["Omega_m"],
    )
    rate_grid = madau_dickinson_rate(
        z_grid,
        FIDUCIALS["gamma"],
        FIDUCIALS["kappa"],
        FIDUCIALS["z_peak"],
        FIDUCIALS["local_merger_rate"],
    )
    integral_mpc3 = jnp.trapezoid(
        rate_grid / (1.0 + z_grid) * dvc_dz_grid,
        z_grid,
    )
    expected = 1e-9 * float(integral_mpc3) / SECONDS_PER_YEAR

    assert float(total_rate) == pytest.approx(expected)


def test_fiducial_weights_cancel_exactly(
    synthetic_importance,
) -> None:
    # The catalog's cached proposal density and reference distance are the same
    # expressions the target forms at FIDUCIALS, so at the fiducial point the
    # log-weights are identically zero and the relative ESS is exactly 1.
    importance, _ = synthetic_importance()
    _, log_weights = _reweight(importance, FIDUCIALS)
    log_weights = np.asarray(log_weights)
    weights = np.exp(log_weights)
    # Exactly, not to tolerance: the two sides are bit-identical expressions.
    # A tolerance here would hide an operation-order change that costs a ulp
    # per weight -- small on its own, but the identity is what several other
    # tests build their exact expectations on.
    np.testing.assert_array_equal(log_weights, np.zeros_like(log_weights))
    assert np.all(np.isfinite(weights))
    rel_ess = float(weights.sum() ** 2 / (weights.size * (weights**2).sum()))
    assert rel_ess == pytest.approx(1.0)
