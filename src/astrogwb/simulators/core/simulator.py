"""The one protocol every simulator implements.

A simulator is a *partial*: built from a metadata record and cost-only settings
(a chunk size, a superbatch), then called on the inputs that vary per item.
Everything the result depends on is either in the metadata, bound at
construction, or an argument of the call -- there is no hidden state.

``__call__`` is the whole interface, and its signature and the layout of its
output ``D`` are the simulator's own contract: a population or spectra simulator
takes a batch of keys and returns one item per key (the population's layout is
flat and ragged), while a catalog simulator takes one key and returns one
catalog, already vectorized over its events. Where a call takes a batch, item
``i`` depends on its own argument alone, up to summation-order bits for
reductions packed across items. Batching is each simulator's own business (a
packed ``segment_sum``, a loop over a compiled stage), not ``jax.vmap``, so it
stays callable eagerly.

A *stochastic* simulator is one whose inputs include a JAX key. Batched keys come
from :func:`~astrogwb.simulators.core.rng.batch_keys`.
"""

from __future__ import annotations

from typing import Protocol

from astrogwb.simulators.core.keys import Keyed

__all__ = ["Simulator"]


class Simulator[**P, D, M: Keyed](Protocol):
    """Called on the inputs that vary; ``D`` is a tree of arrays.

    ``D`` is a per-simulator ``TypedDict`` whose leaves are array-like, so it
    writes as an HDF5 group tree (:func:`~astrogwb.simulators.core.io.write`).
    ``M`` is the metadata record; ``metadata.key()`` names what was simulated.
    """

    @property
    def metadata(self) -> M: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> D: ...
