from __future__ import annotations

import jax

SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


def years_to_seconds(observation_time_yr: float | jax.Array) -> float | jax.Array:
    """Convert an observation time from years to seconds."""
    return observation_time_yr * SECONDS_PER_YEAR
