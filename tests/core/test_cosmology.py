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


def _assert_finite_grad(
    fn: Callable[..., jax.Array], argnums: int, *args: jax.Array
) -> None:
    grad_jnp = jax.grad(fn, argnums=argnums)(*args)
    assert isinstance(grad_jnp, jax.Array)
    assert grad_jnp.shape == args[argnums].shape
    assert jnp.all(jnp.isfinite(grad_jnp))


def _assert_returns_jax_arrays(fn: Callable, *arrays: npt.ArrayLike) -> None:
    """Assert the public contract: ``ArrayLike`` in, ``jax.Array`` out.

    ``jax.typing.ArrayLike`` covers Python scalars, NumPy arrays and JAX
    arrays (not lists), so those are the spellings a caller may legitimately
    pass. None of them may leak a NumPy result back out, and all of them must
    agree numerically.
    """
    spellings: dict[str, tuple] = {
        "numpy": _as_np(*arrays),
        "jax": _as_jnp(*arrays),
    }
    if all(np.ndim(array) == 0 for array in arrays):
        spellings["python"] = tuple(float(np.asarray(a)) for a in arrays)

    reference: tuple[npt.NDArray, ...] | None = None
    for name, args in spellings.items():
        result = fn(*args)
        results = result if isinstance(result, tuple) else (result,)
        assert all(isinstance(r, jax.Array) for r in results), name

        expected_shape = np.broadcast_shapes(*(np.shape(a) for a in arrays))
        assert all(r.shape == expected_shape for r in results), name

        values = tuple(np.asarray(r) for r in results)
        if reference is None:
            reference = values
        else:
            for got, want in zip(values, reference, strict=True):
                np.testing.assert_allclose(got, want, err_msg=name)


@pytest.mark.parametrize("h0", [70])
def test_hubble_distance_returns_jax_array(h0: float) -> None:
    _assert_returns_jax_arrays(hubble_distance, h0)


def test_hubble_distance_grad_wrt_hubble_constant() -> None:
    _assert_finite_grad(hubble_distance, 0, jnp.asarray(70.0))


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

    def test_returns_jax_arrays(
        self, redshift: npt.NDArray, omega_m: npt.NDArray
    ) -> None:
        _assert_returns_jax_arrays(normalized_hubble_parameter, redshift, omega_m)

    def test_jittable(self, redshift: npt.NDArray, omega_m: npt.NDArray) -> None:
        _z, _om = _as_jnp(redshift, omega_m)
        jitted_fn = jax.jit(normalized_hubble_parameter)
        res = jitted_fn(_z, _om)
        res.block_until_ready()

    def test_grad_wrt_omega_m(
        self, redshift: npt.NDArray, omega_m: npt.NDArray
    ) -> None:
        _z, _om = _as_jnp(redshift, omega_m)
        _assert_finite_grad(
            lambda z, om: normalized_hubble_parameter(z, om).sum(), 1, _z, _om
        )

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
    def test_returns_jax_arrays(
        self, redshift: npt.NDArray, hubble_constant, omega_m: npt.NDArray
    ) -> None:
        _assert_returns_jax_arrays(
            distance_and_volume_grid,
            redshift,
            hubble_constant,
            omega_m,
        )

    def test_integer_redshift_grid_matches_float(self, parameters: dict) -> None:
        """An integer grid must promote through float, not truncate the quadrature.

        The Gauss-Legendre nodes and weights are cast to the working dtype. Cast
        to an *integer* redshift dtype they all truncate to zero, and the whole
        integral silently evaluates to zeros -- no warning, no NaN. The same
        grid spelled as integers and as floats must give the same answer.
        """
        integer_grid = np.arange(0, 21)
        float_grid = integer_grid.astype(np.float64)

        _assert_returns_jax_arrays(
            distance_and_volume_grid,
            integer_grid,
            parameters["hubble_constant"],
            parameters["omega_m"],
        )

        from_integers = distance_and_volume_grid(integer_grid, **parameters)
        from_floats = distance_and_volume_grid(float_grid, **parameters)
        for got, want in zip(from_integers, from_floats, strict=True):
            np.testing.assert_allclose(got, want)

        # Pins the specific failure mode: truncated nodes/weights zero the
        # integral, so every distance past z=0 collapses to exactly 0.
        assert np.all(np.asarray(from_integers[0][1:]) > 0.0)

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

    def test_jittable(
        self, redshift: npt.NDArray, hubble_constant: npt.NDArray, omega_m: npt.NDArray
    ) -> None:
        _z, _h, _om = _as_jnp(redshift, hubble_constant, omega_m)
        jitted_fn = jax.jit(distance_and_volume_grid)
        (d, _) = jitted_fn(redshift, hubble_constant=_h, omega_m=_om)
        d.block_until_ready()

    def test_grad_wrt_hubble_constant(
        self, redshift: npt.NDArray, hubble_constant: npt.NDArray, omega_m: npt.NDArray
    ) -> None:
        _z, _h, _om = _as_jnp(redshift, hubble_constant, omega_m)
        _assert_finite_grad(
            lambda z, h, om: distance_and_volume_grid(z, h, om)[0].sum(), 1, _z, _h, _om
        )
        _assert_finite_grad(
            lambda z, h, om: distance_and_volume_grid(z, h, om)[1].sum(), 1, _z, _h, _om
        )

    def test_grad_wrt_omega_m(
        self, redshift: npt.NDArray, hubble_constant: npt.NDArray, omega_m: npt.NDArray
    ) -> None:
        _z, _h, _om = _as_jnp(redshift, hubble_constant, omega_m)
        _assert_finite_grad(
            lambda z, h, om: distance_and_volume_grid(z, h, om)[0].sum(), 2, _z, _h, _om
        )
        _assert_finite_grad(
            lambda z, h, om: distance_and_volume_grid(z, h, om)[1].sum(), 2, _z, _h, _om
        )

    def test_grad_jittable(
        self, redshift: npt.NDArray, hubble_constant: npt.NDArray, omega_m: npt.NDArray
    ) -> None:
        _z, _h, _om = _as_jnp(redshift, hubble_constant, omega_m)
        grad_fn = jax.jit(
            jax.grad(
                lambda z, h, om: distance_and_volume_grid(z, h, om)[0].sum(), argnums=2
            )
        )
        grad_fn(_z, _h, _om).block_until_ready()

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
