"""The simulator protocol, batched keys, content keys and the HDF5 layer.

Nothing in this package imports :mod:`astrogwb.populations`,
:mod:`astrogwb.waveform`, :mod:`astrogwb.gwb` or :mod:`astrogwb.inference`;
``tests/core/test_simulators_core_imports.py`` asserts it.
"""

from astrogwb.simulators.core.io import load, write
from astrogwb.simulators.core.keys import CATALOG_KEY_LENGTH, Keyed, content_key
from astrogwb.simulators.core.rng import batch_keys
from astrogwb.simulators.core.simulator import Simulator
from astrogwb.simulators.core.types import Arrays, Tree

__all__ = [
    "CATALOG_KEY_LENGTH",
    "Arrays",
    "Keyed",
    "Simulator",
    "Tree",
    "batch_keys",
    "content_key",
    "load",
    "write",
]
