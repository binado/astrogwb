from collections.abc import Callable
from functools import wraps

import jax

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
