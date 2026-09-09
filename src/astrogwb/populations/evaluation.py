"""Trace readers used by catalog consistency checks and inference."""

from __future__ import annotations

from astrogwb.populations.base import PopulationTrace

__all__ = [
    "LUMINOSITY_DISTANCE_SITE",
    "REDSHIFT_SITE",
    "TOTAL_MERGER_RATE_SITE",
    "PopulationTrace",
]

#: The redshift site every population must declare. Its density can never be
#: excluded: redshift is the one source parameter the target and the proposal
#: are guaranteed to disagree on.
REDSHIFT_SITE = "redshift"

#: Deterministic distance governing waveform amplitude, in Mpc, shape ``(N,)``.
#: Includes modified GW propagation where the population declares it.
LUMINOSITY_DISTANCE_SITE = "luminosity_distance"

#: Deterministic observer-frame total merger rate, in mergers per second,
#: shape ``()``. A population declares it only when ``params`` carries the
#: physical rate parameters; a proposal density does not need one.
TOTAL_MERGER_RATE_SITE = "total_merger_rate"
