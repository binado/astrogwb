from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from tolerances import assert_close

from astrogwb.cosmology import (
    comoving_distance,
    differential_comoving_volume,
    distance_and_volume_grid,
    hubble_distance,
    luminosity_distance,
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


@pytest.mark.parametrize("redshift", [0.1, 1.0, 5.0, 10.0])
def test_comoving_distance_matches_einstein_de_sitter(redshift: float) -> None:
    """At Omega_m = 1, chi = 2 D_H (1 - 1 / sqrt(1 + z)) in closed form."""
    h0 = 70.0
    expected = 2.0 * hubble_distance(h0) * (1.0 - 1.0 / np.sqrt(1.0 + redshift))
    assert_close(comoving_distance(redshift, h0, 1.0), expected)


def test_pointwise_distances_match_the_grid_helper(parameters) -> None:
    """Both routes agree on the grid nodes, in any order of the points."""
    grid = jnp.linspace(0.0, 10.0, 257)
    grid_distance, grid_volume = distance_and_volume_grid(grid, **parameters)
    shuffled = jax.random.permutation(jax.random.PRNGKey(0), grid.shape[0])
    assert_close(
        luminosity_distance(grid[shuffled], **parameters),
        grid_distance[shuffled],
    )
    assert_close(
        differential_comoving_volume(grid[shuffled], **parameters),
        grid_volume[shuffled],
    )
