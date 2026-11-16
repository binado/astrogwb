from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from astrogwb.cosmology import (
    comoving_distance,
    differential_comoving_volume,
    distance_and_volume_grid,
    hubble_distance,
    hubble_time_gyr,
    lookback_time,
    luminosity_distance,
    normalized_hubble_parameter,
    redshift_at_lookback_time,
)


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


def _as_jnp(*arrs: npt.ArrayLike) -> tuple[jax.Array, ...]:
    return tuple(jnp.asarray(arr) for arr in arrs)


def _assert_finite_grad(
    fn: Callable[..., jax.Array], argnums: int, *args: jax.Array
) -> None:
    grad_jnp = jax.grad(fn, argnums=argnums)(*args)
    assert isinstance(grad_jnp, jax.Array)
    assert grad_jnp.shape == args[argnums].shape
    assert jnp.all(jnp.isfinite(grad_jnp))


class TestNormalizedHubbleParameter:
    @pytest.mark.integration
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
    @pytest.mark.parametrize("output", [0, 1], ids=["distance", "volume"])
    @pytest.mark.parametrize("argnum", [1, 2], ids=["hubble_constant", "omega_m"])
    def test_jitted_gradients_are_finite(
        self,
        redshift: npt.NDArray,
        hubble_constant: npt.NDArray,
        omega_m: npt.NDArray,
        output: int,
        argnum: int,
    ) -> None:
        _z, _h, _om = _as_jnp(redshift, hubble_constant, omega_m)
        _assert_finite_grad(
            jax.jit(lambda z, h, om: distance_and_volume_grid(z, h, om)[output].sum()),
            argnum,
            _z,
            _h,
            _om,
        )

    @pytest.mark.integration
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
        # Reference values from gwmock-pop, which integrates on its own fine
        # internal grid. distance_and_volume_grid uses a 4-point Gauss-Legendre
        # rule per interval, so the residual difference (~1e-9 relative) is the
        # reference's own trapezoid error at this n_grid.
        n_grid = 100_000
        luminosity_distance_theirs = compute_luminosity_distance(
            _z, _h, _om, n_grid=n_grid
        )
        differential_comoving_volume_theirs = compute_differential_comoving_volume(
            _z, _h, _om, n_grid=n_grid
        )
        np.testing.assert_allclose(
            luminosity_distance, luminosity_distance_theirs, rtol=1e-8
        )
        np.testing.assert_allclose(
            differential_comoving_volume, differential_comoving_volume_theirs, rtol=1e-8
        )


def test_lookback_time_derivative_is_the_integrand(parameters) -> None:
    """The closed form differentiates to dt_L/dz = t_H / ((1 + z) E(z))."""
    redshift = jnp.linspace(0.0, 20.0, 64)
    derivative = jax.vmap(jax.grad(lambda z: lookback_time(z, **parameters)))(redshift)
    expected = hubble_time_gyr(parameters["hubble_constant"]) / (
        (1.0 + redshift) * normalized_hubble_parameter(redshift, parameters["omega_m"])
    )
    np.testing.assert_allclose(derivative, expected, rtol=1e-12)


def test_lookback_time_tends_to_the_age_of_the_universe(parameters) -> None:
    """Planck-like parameters give an age near 13.8 Gyr (no radiation term)."""
    assert lookback_time(0.0, **parameters) == 0.0
    np.testing.assert_allclose(lookback_time(1e8, **parameters), 13.81, atol=0.01)


def test_redshift_at_lookback_time_inverts_lookback_time(parameters) -> None:
    redshift = jnp.linspace(0.0, 50.0, 128)
    round_tripped = redshift_at_lookback_time(
        lookback_time(redshift, **parameters), **parameters
    )
    np.testing.assert_allclose(round_tripped, redshift, rtol=1e-10, atol=1e-13)


def test_redshift_at_lookback_time_is_infinite_beyond_the_age(parameters) -> None:
    """Past the age the redshift is inf, with a finite gradient everywhere."""
    times = jnp.array([1.0, 14.0, 20.0])
    redshift = redshift_at_lookback_time(times, **parameters)
    assert np.isfinite(redshift[0])
    assert np.all(np.isposinf(redshift[1:]))
    gradient = jax.grad(
        lambda t: jnp.sum(
            jnp.where(t < 13.0, redshift_at_lookback_time(t, **parameters), 0.0)
        )
    )(times)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("redshift", [0.1, 1.0, 5.0, 10.0])
def test_comoving_distance_matches_einstein_de_sitter(redshift: float) -> None:
    """At Omega_m = 1, chi = 2 D_H (1 - 1 / sqrt(1 + z)) in closed form."""
    h0 = 70.0
    expected = 2.0 * hubble_distance(h0) * (1.0 - 1.0 / np.sqrt(1.0 + redshift))
    np.testing.assert_allclose(
        comoving_distance(redshift, h0, 1.0), expected, rtol=1e-9
    )


def test_pointwise_distances_match_the_grid_helper(parameters) -> None:
    """Both routes agree on the grid nodes, in any order of the points."""
    grid = jnp.linspace(0.0, 10.0, 257)
    grid_distance, grid_volume = distance_and_volume_grid(grid, **parameters)
    shuffled = jax.random.permutation(jax.random.PRNGKey(0), grid.shape[0])
    np.testing.assert_allclose(
        luminosity_distance(grid[shuffled], **parameters),
        grid_distance[shuffled],
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        differential_comoving_volume(grid[shuffled], **parameters),
        grid_volume[shuffled],
        rtol=1e-8,
        atol=1e-4,
    )
