"""Protocol for importance-weighted merger-rate callbacks.

The reference realization in
:mod:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation`
is built from a :class:`~astrogwb.importance.population.PopulationFn`; this
protocol is the catalog-facing shape the NumPyro models consume.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import jax


class MergerRateAndLogWeightsFn(Protocol):
    """Calculate a total merger rate and catalog importance weights."""

    def __call__(
        self,
        params: Mapping[str, Any],
        samples: Mapping[str, jax.Array],
    ) -> tuple[float | jax.Array, jax.Array]: ...
