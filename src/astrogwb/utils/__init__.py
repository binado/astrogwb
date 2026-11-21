from collections.abc import Callable, Mapping, Sequence
from functools import wraps

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpy.polynomial.legendre import leggauss

from astrogwb import __version__
from astrogwb.constants import SECONDS_PER_YEAR

type PairName = tuple[str, str]


def require_current_version(version: str) -> None:
    """Raise unless a record's ``version`` is the installed astrogwb version.

    Parameters
    ----------
    version
        The version a metadata record was made with.

    Raises
    ------
    ValueError
        If ``version`` is not :data:`astrogwb.__version__`.
    """
    if version != __version__:
        raise ValueError(
            f"metadata is for astrogwb {version}, but {__version__} is installed; "
            "regenerate it"
        )


def years_to_seconds[T: (float, jax.Array)](observation_time_yr: T) -> T:
    """Convert an observation time from years to seconds.

    Parameters
    ----------
    observation_time_yr
        Observation time in years, a float or a JAX array.

    Returns
    -------
    float or jax.Array
        The same time in seconds, of the type of the argument.
    """
    return observation_time_yr * SECONDS_PER_YEAR


def require_x64[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Raise unless JAX x64 mode is enabled at call time.

    Parameters
    ----------
    function
        Function to wrap.
    """

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not jax.config.x64_enabled:
            raise RuntimeError(
                f"{wrapper.__name__} requires JAX x64 mode because realistic "
                "strain spectra and intensities underflow in float32; "
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

    Parameters
    ----------
    y
        Values to integrate.
    x
        Integration abscissa.
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

    ``n`` is static; the result is jit-safe and differentiable in ``a`` and ``b``.

    Parameters
    ----------
    a, b
        Integration bounds, broadcast against each other.
    n
        Number of quadrature nodes per interval.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(points, weights)``, each of shape
        ``jnp.broadcast_shapes(a.shape, b.shape) + (n,)``, with dtype
        ``jnp.result_type(a, b, float)``.
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


def stack_pairs(
    values: Mapping[PairName, ArrayLike], pairs: Sequence[PairName]
) -> jax.Array:
    """Stack pair-keyed arrays along a new pair axis, in ``pairs`` order.

    Parameters
    ----------
    values
        One array per pair, all of the same shape ``(..., F)``.
    pairs
        The pairs, in the order of the stacked axis.

    Returns
    -------
    jax.Array
        The stacked arrays, ``(..., P, F)``.

    Raises
    ------
    ValueError
        If the keys of ``values`` are not exactly ``pairs``, or the arrays do
        not share one shape with a trailing frequency axis.
    """
    if set(values) != set(pairs):
        raise ValueError(
            f"expected one array per sorted detector pair {list(pairs)}, "
            f"got {sorted(values)}"
        )
    arrays = [jnp.asarray(values[pair]) for pair in pairs]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1 or not arrays[0].shape:
        raise ValueError(
            f"per-pair arrays must share a shape (..., F), got {sorted(shapes)}"
        )
    return jnp.stack(arrays, axis=-2)


def unstack_pairs(
    array: ArrayLike, pairs: Sequence[PairName]
) -> dict[PairName, jax.Array]:
    """Split the pair axis of ``array``, the inverse of :func:`stack_pairs`.

    Parameters
    ----------
    array
        Stacked values, ``(..., P, F)``.
    pairs
        The pair of each entry of the pair axis.

    Returns
    -------
    dict[tuple[str, str], jax.Array]
        One ``(..., F)`` array per pair.

    Raises
    ------
    ValueError
        If the pair axis does not have one entry per pair.
    """
    array = jnp.asarray(array)
    if array.ndim < 2 or array.shape[-2] != len(pairs):
        raise ValueError(
            f"the pair axis of an array of shape {array.shape} does not match "
            f"the {len(pairs)} pairs"
        )
    return {pair: array[..., index, :] for index, pair in enumerate(pairs)}
