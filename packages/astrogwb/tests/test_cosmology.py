from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from astrogwb.cosmology import (
    distance_and_volume_grid,
    hubble_distance,
    normalized_hubble_parameter,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def parameters():
    return {"hubble_constant": 67.66, "omega_m": 0.3096}


@pytest.fixture
def redshift():
    return np.linspace(0, 20, 128)


@pytest.fixture
def hubble_constant():
    return np.linspace(60, 80, 8).reshape(-1, 1)


@pytest.fixture
def omega_m():
    return np.linspace(0.2, 0.4, 8).reshape(-1, 1)


def _as_np(*arrs: npt.ArrayLike) -> tuple[npt.NDArray, ...]:
    return tuple(np.asarray(arr) for arr in arrs)


def _as_jnp(*arrs: npt.ArrayLike) -> tuple[jax.Array, ...]:
    return tuple(jnp.asarray(arr) for arr in arrs)


def _test_array_backend_scalar_input(fn: Callable, x: float) -> None:
    result = fn(x)
    assert isinstance(result, float)

    x_as_np = np.asarray(x)
    result_np = fn(x_as_np)
    assert isinstance(result_np, np.ndarray)
    assert result_np.shape == x_as_np.shape
    assert result_np.ndim == 0

    x_as_jnp = jnp.asarray(x)
    result_jnp = fn(x_as_jnp)
    assert isinstance(result_jnp, jax.Array)
    assert result_jnp.shape == x_as_jnp.shape
    assert result_jnp.ndim == 0


def _test_array_backend_array_input(fn: Callable, *arrays: npt.ArrayLike):
    arrays_np = _as_np(*arrays)
    result_np = fn(*arrays_np)
    results_np = result_np if isinstance(result_np, tuple) else (result_np,)
    assert all(isinstance(r, np.ndarray) for r in results_np)

    arrays_jnp = _as_jnp(*arrays)
    result_jnp = fn(*arrays_jnp)
    results_jnp = result_jnp if isinstance(result_jnp, tuple) else (result_jnp,)
    assert all(isinstance(r, jax.Array) for r in results_jnp)


@pytest.mark.parametrize("h0", [70])
def test_hubble_distance_preserves_array_backend(h0: float) -> None:
    _test_array_backend_scalar_input(hubble_distance, h0)


class TestNormalizedHubbleParameter:
    @pytest.mark.parametrize(
        "omega_m",
        [
            0.3,
            np.asarray([0.3, 0.4, 0.5]).reshape(-1, 1),
            np.asarray([0.3, 0.4, 0.5]).reshape(-1, 1, 1),
        ],
    )
    def test_broadcasting(self, redshift: npt.NDArray, omega_m: npt.ArrayLike) -> None:
        _omega_m = np.asarray(omega_m)
        ez = normalized_hubble_parameter(redshift, _omega_m)
        expected_shape = np.broadcast_shapes(redshift.shape, _omega_m.shape)
        assert ez.shape == expected_shape

    def test_preserves_array_backend(
        self, redshift: npt.NDArray, omega_m: npt.NDArray
    ) -> None:
        _test_array_backend_array_input(normalized_hubble_parameter, redshift, omega_m)

    def test_matches_gwmockpop(
        self, redshift: npt.NDArray, omega_m: npt.NDArray
    ) -> None:
        from gwmock_pop.cosmology.flat_lambda_cdm import (
            compute_normalized_hubble_parameter,
        )

        _redshift, _omega_m = _as_jnp(redshift, omega_m)
        ours = normalized_hubble_parameter(_redshift, _omega_m)
        theirs = compute_normalized_hubble_parameter(_redshift, _omega_m)
        np.testing.assert_allclose(ours, theirs)


class TestDistanceAndVolumeGrid:
    def test_preserves_array_backend(
        self, redshift: npt.NDArray, hubble_constant, omega_m: npt.NDArray
    ) -> None:
        _test_array_backend_array_input(
            lambda z, h, om: distance_and_volume_grid(z, hubble_constant=h, omega_m=om),
            redshift,
            hubble_constant,
            omega_m,
        )

    @pytest.mark.parametrize(
        "hubble_constant,omega_m",
        [
            (70, 0.3),
            (70, np.asarray([0.3, 0.4]).reshape(-1, 1)),
            (np.asarray([70, 71]).reshape(-1, 1), 0.3),
            (
                np.asarray([70, 71]).reshape(-1, 1),
                np.asarray([0.3, 0.4]).reshape(-1, 1),
            ),
        ],
    )
    def test_broadcasting(
        self,
        redshift: npt.NDArray,
        hubble_constant: npt.ArrayLike,
        omega_m: npt.ArrayLike,
    ) -> None:
        _h, _om = _as_np(hubble_constant, omega_m)
        res = distance_and_volume_grid(redshift, hubble_constant=_h, omega_m=_om)
        shape = np.broadcast(redshift, _h, _om).shape
        assert all(r.shape == shape for r in res)

    @pytest.mark.parametrize("hubble_constant,omega_m", [(70, 0.3)])
    def test_matches_gwmock_pop(
        self, redshift: npt.NDArray, hubble_constant: npt.NDArray, omega_m: npt.NDArray
    ) -> None:
        from gwmock_pop.cosmology.flat_lambda_cdm import (
            compute_differential_comoving_volume,
            compute_luminosity_distance,
        )

        _z, _h, _om = _as_jnp(redshift, hubble_constant, omega_m)
        luminosity_distance, differential_comoving_volume = distance_and_volume_grid(
            _z,
            hubble_constant=_h,
            omega_m=_om,
        )
        n_grid = _z.shape[-1]
        luminosity_distance_theirs = compute_luminosity_distance(
            _z, _h, _om, n_grid=n_grid
        )
        differential_comoving_volume_theirs = compute_differential_comoving_volume(
            _z, _h, _om, n_grid=n_grid
        )
        np.testing.assert_allclose(luminosity_distance, luminosity_distance_theirs)
        np.testing.assert_allclose(
            differential_comoving_volume, differential_comoving_volume_theirs
        )
