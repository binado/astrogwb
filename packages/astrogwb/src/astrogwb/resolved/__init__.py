"""Resolved (compact-binary) signal analysis built on the stochastic core.

Where :mod:`astrogwb.gwb` treats the gravitational-wave background as a
stochastic field, this subpackage resolves individual compact-binary events:
it generates time-domain waveforms, projects them onto a detector network,
and contracts them against the network noise.
"""

from .snr import network_optimal_snr

__all__ = ["network_optimal_snr"]
