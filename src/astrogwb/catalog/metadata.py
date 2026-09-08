"""Generation provenance for one catalog.

Distinct from the *density spec* a catalog carries. The spec -- registry name,
model construction settings, generating hyperparameters, included density
factors -- fully determines the law the samples follow, and lives on
:class:`~astrogwb.catalog.Catalog`. What is left here is provenance: the seed
and sample count a draw is reproducible from, and free-form scalar notes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = ["PopulationMetadata", "ScalarProvenance"]

ScalarProvenance = str | int | float


@dataclass(frozen=True, slots=True)
class PopulationMetadata:
    """Population identity, draw settings, and scalar provenance."""

    name: str
    seed: int
    num_samples: int
    source_type: str | None = None
    provenance: Mapping[str, ScalarProvenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if isinstance(self.num_samples, bool) or not isinstance(self.num_samples, int):
            raise TypeError("num_samples must be an int")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        for key, value in self.provenance.items():
            if not isinstance(key, str):
                raise TypeError("provenance names must be strings")
            if isinstance(value, bool) or not isinstance(value, str | int | float):
                raise TypeError(
                    f"provenance[{key!r}] must be a str, non-boolean int, or "
                    f"float scalar, got {type(value).__name__}"
                )
