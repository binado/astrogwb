from collections.abc import Callable, Mapping
from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from numpy.polynomial.legendre import leggauss

from astrogwb.constants import SECONDS_PER_YEAR


def years_to_seconds(observation_time_yr: float) -> float:
    """Convert an observation time from years to seconds."""
    return observation_time_yr * SECONDS_PER_YEAR


def array_dict_shape(parameters: Mapping[str, ArrayLike]) -> tuple[int, ...]:
    """Return the common shape of arrays in ``parameters``.

    Every value must have exactly the same shape. Scalar values are therefore
    valid and return ``()``; callers that require a particular rank should
    validate it separately.
    """
    shapes = [(name, np.shape(values)) for name, values in parameters.items()]
    if not shapes:
        raise ValueError("parameters must contain at least one array")

    reference_name, reference_shape = shapes[0]
    for name, shape in shapes[1:]:
        if shape != reference_shape:
            raise ValueError(
                "parameter arrays must have matching shapes; "
                f"{reference_name!r} has shape {reference_shape}, "
                f"but {name!r} has shape {shape}"
            )
    return tuple(reference_shape)


def require_x64[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Raise unless JAX x64 mode is enabled at call time."""

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not jax.config.x64_enabled:
            raise RuntimeError(
                f"{wrapper.__name__} requires JAX x64 mode because realistic "
                "strain spectra and polarization powers underflow in float32; "
                "call jax.config.update('jax_enable_x64', True) before "
                "creating arrays"
            )
        return function(*args, **kwargs)

    return wrapper


def cumulative_trapezoid(y: jax.Array, x: jax.Array) -> jax.Array:
    """Cumulatively integrate ``y`` against the 1-D abscissa ``x``.

    Integrates along the trailing axis of ``y`` with the trapezoid rule and
    broadcasts over any leading batch dimensions. The result has the shape
    of ``y`` and starts at zero along the integrated axis.
    """
    dx = jnp.diff(x)
    segments = 0.5 * (y[..., :-1] + y[..., 1:]) * dx
    zeros = jnp.zeros(y.shape[:-1] + (1,), dtype=y.dtype)
    return jnp.concatenate([zeros, jnp.cumsum(segments, axis=-1)], axis=-1)


def gauss_legendre_nodes_weights(
    a: ArrayLike, b: ArrayLike, n: int
) -> tuple[jax.Array, jax.Array]:
    r"""Map an ``n``-point Gauss-Legendre rule onto one or many intervals.

    An ``n``-point rule integrates polynomials of degree ``2 * n - 1``
    exactly. ``a`` and ``b`` may be scalars or arrays; both returned arrays
    have the broadcast shape of ``a`` and ``b`` followed by a trailing ``n``
    axis, so a single call covers every interval of a grid.

    The returned weights already carry the affine Jacobian ``(b - a) / 2`` of
    each interval, so the integral of ``f`` over every interval is

    .. code-block:: python

        points, weights = gauss_legendre_nodes_weights(a, b, n)
        integral = jnp.sum(weights * f(points), axis=-1)

    ``n`` is a static Python integer and is never extracted from a traced
    value, so this helper is safe to call inside jitted code, and it is
    differentiable in ``a`` and ``b``.

    Parameters
    ----------
    a, b:
        Integration bounds, broadcast against each other.
    n:
        Number of quadrature nodes per interval.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(points, weights)``, each of shape
        ``jnp.broadcast_shapes(a.shape, b.shape) + (n,)``. The dtype is
        ``jnp.result_type(a, b, float)``: float32 bounds stay float32, integer
        bounds promote to floating point instead of truncating the nodes and
        weights to zeros, and Python floats need JAX x64 mode for float64.
    """
    dtype = jnp.result_type(a, b, float)
    host_nodes, host_weights = leggauss(n)
    nodes = jnp.asarray(host_nodes, dtype=dtype)
    weights = jnp.asarray(host_weights, dtype=dtype)
    a = jnp.asarray(a, dtype=dtype)
    b = jnp.asarray(b, dtype=dtype)
    midpoint = 0.5 * (a + b)
    half_width = 0.5 * (b - a)
    return (
        midpoint[..., None] + half_width[..., None] * nodes,
        half_width[..., None] * weights,
    )
