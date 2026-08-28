from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astrogwb.cosmology import (
    MPC_IN_METERS,
    distance_and_volume_grid,
    hubble_constant_si,
    hubble_distance,
    normalized_hubble_parameter,
)
from gwmock_pop.cosmology.flat_lambda_cdm import (
    compute_differential_comoving_volume,
    compute_luminosity_distance,
    compute_normalized_hubble_parameter,
)

jax.config.update("jax_enable_x64", True)

_FIDUCIALS = {"H0": 67.66, "Omega_m": 0.3096}


def test_hubble_constant_si_converts_km_s_mpc_to_si() -> None:
    np.testing.assert_allclose(
        hubble_constant_si(67.74),
        67.74 * 1000.0 / MPC_IN_METERS,
    )
    np.testing.assert_allclose(
        hubble_constant_si(67.66),
        67.66 * 1000.0 / MPC_IN_METERS,
    )


def test_hubble_distance_preserves_array_backend() -> None:
    h0 = np.array([67.66, 67.74])

    distance = hubble_distance(h0)
    assert isinstance(distance, np.ndarray)
    np.testing.assert_allclose(distance, [hubble_distance(float(x)) for x in h0])
    assert isinstance(hubble_distance(jnp.asarray(h0)), jax.Array)


def test_normalized_hubble_parameter_matches_gwmock_pop() -> None:
    redshift = np.array([0.0, 0.3, 1.0, 2.7, 8.0, 20.0])

    e_z = normalized_hubble_parameter(redshift, _FIDUCIALS["Omega_m"])
    assert isinstance(e_z, np.ndarray)
    assert float(e_z[0]) == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_allclose(
        e_z,
        np.asarray(
            compute_normalized_hubble_parameter(
                redshift=jnp.asarray(redshift),
                omega_m=jnp.asarray(_FIDUCIALS["Omega_m"]),
            )
        ),
        rtol=1e-12,
    )

    e_z_jax = normalized_hubble_parameter(jnp.asarray(redshift), _FIDUCIALS["Omega_m"])
    assert isinstance(e_z_jax, jax.Array)
    np.testing.assert_allclose(np.asarray(e_z_jax), e_z, rtol=1e-12)


def test_distance_and_volume_grid_vanishes_at_redshift_zero() -> None:
    redshift = jnp.array([0.0, 0.3, 1.0, 2.7, 8.0, 20.0])
    luminosity_distance, differential_comoving_volume = distance_and_volume_grid(
        _FIDUCIALS, redshift
    )
    luminosity_distance = np.asarray(luminosity_distance)
    differential_comoving_volume = np.asarray(differential_comoving_volume)

    assert float(luminosity_distance[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(differential_comoving_volume[0]) == pytest.approx(0.0, abs=1e-9)
    comoving_distance = luminosity_distance / (1.0 + np.asarray(redshift))
    assert np.all(np.diff(comoving_distance) > 0)
    assert not np.allclose(comoving_distance, comoving_distance[-1])


def test_distance_and_volume_grid_broadcasts_batched_parameters() -> None:
    redshift = np.linspace(0.0, 3.0, 128)
    params = {
        "H0": np.array([[67.66], [70.0]]),
        "Omega_m": np.array([[0.3096], [0.27]]),
    }

    luminosity_distance, differential_comoving_volume = distance_and_volume_grid(
        params, redshift
    )

    assert luminosity_distance.shape == (2, redshift.size)
    assert differential_comoving_volume.shape == (2, redshift.size)
    for batch_index in range(2):
        expected_luminosity_distance, expected_differential_comoving_volume = (
            distance_and_volume_grid(
                {
                    "H0": float(params["H0"][batch_index, 0]),
                    "Omega_m": float(params["Omega_m"][batch_index, 0]),
                },
                redshift,
            )
        )
        np.testing.assert_allclose(
            luminosity_distance[batch_index], expected_luminosity_distance
        )
        np.testing.assert_allclose(
            differential_comoving_volume[batch_index],
            expected_differential_comoving_volume,
        )


def test_distance_and_volume_grid_integrates_along_last_axis() -> None:
    redshift = np.stack(
        [
            np.linspace(0.0, 3.0, 128),
            np.linspace(0.0, 2.0, 128),
        ]
    )
    params = {
        "H0": np.array([[67.66], [70.0]]),
        "Omega_m": np.array([[0.3096], [0.27]]),
    }

    luminosity_distance, differential_comoving_volume = distance_and_volume_grid(
        params, redshift
    )

    assert luminosity_distance.shape == redshift.shape
    assert differential_comoving_volume.shape == redshift.shape
    for batch_index in range(redshift.shape[0]):
        expected_luminosity_distance, expected_differential_comoving_volume = (
            distance_and_volume_grid(
                {
                    "H0": float(params["H0"][batch_index, 0]),
                    "Omega_m": float(params["Omega_m"][batch_index, 0]),
                },
                redshift[batch_index],
            )
        )
        np.testing.assert_allclose(
            luminosity_distance[batch_index], expected_luminosity_distance
        )
        np.testing.assert_allclose(
            differential_comoving_volume[batch_index],
            expected_differential_comoving_volume,
        )


def test_distance_and_volume_grid_batched_numpy_matches_jax() -> None:
    redshift = np.stack(
        [
            np.linspace(0.0, 3.0, 128),
            np.linspace(0.0, 2.0, 128),
        ]
    )
    params = {
        "H0": np.array([[67.66], [70.0]]),
        "Omega_m": np.array([[0.3096], [0.27]]),
    }

    d_l_np, dvc_np = distance_and_volume_grid(params, redshift)
    d_l_jax, dvc_jax = distance_and_volume_grid(
        {name: jnp.asarray(value) for name, value in params.items()},
        jnp.asarray(redshift),
    )

    np.testing.assert_allclose(d_l_np, np.asarray(d_l_jax), rtol=1e-12)
    np.testing.assert_allclose(dvc_np, np.asarray(dvc_jax), rtol=1e-12)


def test_distance_and_volume_grid_matches_low_redshift_limit() -> None:
    redshift = jnp.array([0.0, 1.0e-3])
    luminosity_distance, _ = distance_and_volume_grid(_FIDUCIALS, redshift)
    hubble = float(hubble_distance(_FIDUCIALS["H0"]))
    np.testing.assert_allclose(
        luminosity_distance[1],
        hubble * 1.0e-3 * 1.001,
        rtol=1e-3,
    )


def test_distance_and_volume_grid_offset_matches_from_zero_prefix() -> None:
    # The virtual z = 0 knot keeps grids that start above zero equal to the
    # matching tail of a from-zero grid, rather than integrating only from z_min.
    from_zero = jnp.array([0.0, 0.5, 1.2, 3.0])
    offset = jnp.array([0.5, 1.2, 3.0])
    d_l_from_zero, dvc_from_zero = distance_and_volume_grid(_FIDUCIALS, from_zero)
    d_l_offset, dvc_offset = distance_and_volume_grid(_FIDUCIALS, offset)
    np.testing.assert_allclose(d_l_offset, d_l_from_zero[1:])
    np.testing.assert_allclose(dvc_offset, dvc_from_zero[1:])


def test_distance_and_volume_grid_numpy_backend_matches_jax() -> None:
    redshift = np.array([0.0, 0.3, 1.0, 2.7, 8.0, 20.0])

    d_l_np, dvc_np = distance_and_volume_grid(_FIDUCIALS, redshift)
    assert isinstance(d_l_np, np.ndarray)
    assert isinstance(dvc_np, np.ndarray)

    d_l_jax, dvc_jax = distance_and_volume_grid(_FIDUCIALS, jnp.asarray(redshift))
    assert isinstance(d_l_jax, jax.Array)
    assert isinstance(dvc_jax, jax.Array)
    np.testing.assert_allclose(d_l_np, np.asarray(d_l_jax), rtol=1e-12)
    np.testing.assert_allclose(dvc_np, np.asarray(dvc_jax), rtol=1e-12)


def test_distance_and_volume_grid_matches_gwmock_pop_fine_grid() -> None:
    # gwmock_pop integrates c / H(z) with its own dense trapezoid, so only
    # agreement up to integration error is expected.
    redshift = np.linspace(0.0, 3.0, 4096)
    d_l, dvc = distance_and_volume_grid(_FIDUCIALS, redshift)

    d_l_ref = np.asarray(
        compute_luminosity_distance(
            redshift=jnp.asarray(redshift),
            hubble_constant=jnp.asarray(_FIDUCIALS["H0"]),
            omega_m=jnp.asarray(_FIDUCIALS["Omega_m"]),
        )
    )
    dvc_ref = np.asarray(
        compute_differential_comoving_volume(
            redshift=jnp.asarray(redshift),
            hubble_constant=jnp.asarray(_FIDUCIALS["H0"]),
            omega_m=jnp.asarray(_FIDUCIALS["Omega_m"]),
        )
    )
    np.testing.assert_allclose(d_l[1:], d_l_ref[1:], rtol=1e-4)
    np.testing.assert_allclose(dvc[1:], dvc_ref[1:], rtol=1e-4)


def test_distance_and_volume_grid_jit_matches_eager() -> None:
    # The NUTS model traces this with sampled params: traced params and grid
    # must give the same values as eager evaluation.
    redshift = jnp.linspace(0.0, 3.0, 128)

    d_l_traced, dvc_traced = jax.jit(distance_and_volume_grid)(_FIDUCIALS, redshift)
    d_l, dvc = distance_and_volume_grid(_FIDUCIALS, redshift)
    np.testing.assert_allclose(np.asarray(d_l_traced), np.asarray(d_l))
    np.testing.assert_allclose(np.asarray(dvc_traced), np.asarray(dvc))
