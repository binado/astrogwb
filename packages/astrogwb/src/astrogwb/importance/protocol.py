"""Protocol for importance-weighted merger-rate callbacks."""

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
