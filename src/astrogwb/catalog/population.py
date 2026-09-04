"""Population identity, provenance, and graph-based population simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["PopulationMetadata", "simulate_population"]

ScalarProvenance = str | int | float


@dataclass(frozen=True, slots=True)
class PopulationMetadata:
    """Population identity, simulation settings, and scalar provenance."""

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


def simulate_population(
    config: Mapping[str, Any] | str | Path,
    *,
    metadata: PopulationMetadata,
) -> dict[str, NDArray[Any]]:
    """Draw source arrays from a mapping or graph configuration file."""
    try:
        from gwmock_pop import GraphSimulator
    except ImportError as error:
        raise ImportError(
            "Population simulation requires the optional 'gwmock-pop' dependency; "
            "install it with `pip install astrogwb[simulation]`."
        ) from error

    if isinstance(config, Mapping):
        simulator = GraphSimulator(
            cast("dict[str, Any]", config),
            source_type=metadata.source_type,
            seed=metadata.seed,
        )
    else:
        simulator = GraphSimulator.from_config_file(
            config, source_type=metadata.source_type, seed=metadata.seed
        )
    return {
        name: np.asarray(values)
        for name, values in simulator.simulate(metadata.num_samples).items()
    }
