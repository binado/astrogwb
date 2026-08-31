from collections.abc import Callable
from functools import wraps

import jax
import jax.numpy as jnp

SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


def years_to_seconds(observation_time_yr: float) -> float:
    """Convert an observation time from years to seconds."""
    return observation_time_yr * SECONDS_PER_YEAR


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
