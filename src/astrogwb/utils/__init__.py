from collections.abc import Callable, Mapping
from functools import cache, wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

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


@cache
def gauss_legendre_rule(
    order: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return cached host-side float64 nodes and weights on ``[-1, 1]``.

    The rule is built on the host with :func:`numpy.polynomial.legendre.leggauss`
    and memoized on ``order``, so repeated tracing of the same quadrature order
    costs nothing after the first call.
    """
    nodes, weights = leggauss(order)
    return nodes.astype(np.float64), weights.astype(np.float64)


def mapped_gauss_legendre_rule(
    order: int,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    dtype: DTypeLike = jnp.float64,
) -> tuple[jax.Array, jax.Array]:
    r"""Map an ``order``-point Gauss-Legendre rule onto one or many intervals.

    An ``order``-point rule integrates polynomials of degree ``2 * order - 1``
    exactly. ``lower`` and ``upper`` may be scalars or arrays; both returned
    arrays have the broadcast shape of ``lower`` and ``upper`` followed by a
    trailing ``order`` axis, so a single call covers every interval of a grid.

    The returned weights already carry the affine Jacobian ``(upper - lower) / 2``
    of each interval, so the integral of ``f`` over every interval is

    .. code-block:: python

        points, weights = mapped_gauss_legendre_rule(order, lower, upper)
        integral = jnp.sum(weights * f(points), axis=-1)

    ``order`` is a static Python integer and is never extracted from a traced
    value, so this helper is safe to call inside jitted code.

    Parameters
    ----------
    order:
        Number of quadrature nodes per interval.
    lower, upper:
        Integration bounds, broadcast against each other.
    dtype:
        Working floating-point dtype. Defaults to ``float64``, which requires
        JAX x64 mode. Callers that must also work in float32, or that accept
        integer bounds, should pass ``jnp.result_type(bounds, float)``: the
        bounds are cast to ``dtype`` here, so an integer grid promotes instead
        of truncating the nodes and weights to zeros.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(points, weights)``, each of shape
        ``jnp.broadcast_shapes(lower.shape, upper.shape) + (order,)``.
    """
    host_nodes, host_weights = gauss_legendre_rule(order)
    nodes = jnp.asarray(host_nodes, dtype=dtype)
    weights = jnp.asarray(host_weights, dtype=dtype)
    lower = jnp.asarray(lower, dtype=dtype)
    upper = jnp.asarray(upper, dtype=dtype)
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return (
        midpoint[..., None] + half_width[..., None] * nodes,
        half_width[..., None] * weights,
    )
