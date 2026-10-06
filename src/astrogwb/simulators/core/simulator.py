"""The one protocol every simulator implements.

A simulator is a *partial*: built from a metadata record and cost-only settings
(a chunk size, a superbatch), then called on the inputs that vary per item.
Everything the result depends on is either in the metadata, bound at
construction, or an argument of the call -- there is no hidden state.

``simulate_batch`` is semantically ``vmap(simulate)``::

    simulate_batch(*batched)[i] == simulate(*(arg[i] for arg in batched))

up to summation-order bits for reductions packed across items. Every call
argument is batched along a leading axis; anything static is bound in the
constructor. It is a separate method, not ``jax.vmap``, so each simulator
implements batching its own way (a packed ``segment_sum``, a loop over a
compiled stage) and stays callable eagerly.

A *stochastic* simulator is one whose inputs include a JAX key. Batched keys come
from :func:`~astrogwb.simulators.core.rng.batch_keys`. Layouts that are ragged
across the batch (flat sources plus counts) are documented per simulator.
"""

from __future__ import annotations

from typing import Protocol

from astrogwb.simulators.core.keys import Keyed

__all__ = ["Simulator"]


class Simulator[**P, D, M: Keyed](Protocol):
    """``simulate`` one item, or ``simulate_batch`` many; ``D`` is a tree of arrays.

    ``D`` is a per-simulator ``TypedDict`` whose leaves are array-like, so it
    writes as an HDF5 group tree (:func:`~astrogwb.simulators.core.io.write`).
    ``M`` is the metadata record; ``metadata.key()`` names what was simulated.
    """

    @property
    def metadata(self) -> M: ...

    def simulate(self, *args: P.args, **kwargs: P.kwargs) -> D: ...

    def simulate_batch(self, *args: P.args, **kwargs: P.kwargs) -> D: ...
