"""Astrophysical stochastic gravitational-wave background utilities.

``__version__`` is read from the installed distribution, so ``pyproject.toml``
stays its single source. It is part of every catalog's cache key: bump it
whenever a change alters what a population draw or a waveform generator
produces, or a cached catalog built by the old code keeps being served.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astrogwb")
except PackageNotFoundError:  # pragma: no cover - only in an uninstalled tree
    __version__ = "0+unknown"

__all__ = ["__version__"]
