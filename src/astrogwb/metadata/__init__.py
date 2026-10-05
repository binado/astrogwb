"""Leaf records shared by every artifact's metadata.

Only :class:`PriorSpec` lives here: it imports nothing from the physics layers,
so the records that embed it -- the population, waveform, catalog and spectra
metadata -- can each live beside the code they describe
(:mod:`astrogwb.populations.metadata`, :mod:`astrogwb.waveform.metadata`,
:mod:`astrogwb.simulators`).

Importing those records runs their parent package, which imports JAX. That is
allowed; *initializing* the XLA backend is not, and
``tests/core/test_metadata_imports.py`` asserts it stays uninitialized.
"""

from astrogwb.metadata.prior import PriorSpec

__all__ = ["PriorSpec"]
