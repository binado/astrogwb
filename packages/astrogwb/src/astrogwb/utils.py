SECONDS_PER_YEAR: float = 365.25 * 24.0 * 3600.0


def years_to_seconds(observation_time_yr: float) -> float:
    """Convert an observation time from years to seconds."""
    return observation_time_yr * SECONDS_PER_YEAR
