"""Population identity, provenance, and graph-based population simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PopulationMetadata",
    "simulate_population",
    "simulate_population_mixture",
]

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


GraphConfig = Mapping[str, Any] | str | Path


def _graph_simulator(config: GraphConfig, *, source_type: str | None, seed: int) -> Any:
    """Build one gwmock GraphSimulator from a mapping or a config file path."""
    try:
        from gwmock_pop import GraphSimulator
    except ImportError as error:
        raise ImportError(
            "Population simulation requires the optional 'gwmock-pop' dependency; "
            "install it with `pip install astrogwb[simulation]`."
        ) from error

    if isinstance(config, Mapping):
        return GraphSimulator(
            cast("dict[str, Any]", config), source_type=source_type, seed=seed
        )
    return GraphSimulator.from_config_file(config, source_type=source_type, seed=seed)


def simulate_population(
    config: GraphConfig,
    *,
    metadata: PopulationMetadata,
) -> dict[str, NDArray[Any]]:
    """Draw source arrays from a mapping or graph configuration file."""
    simulator = _graph_simulator(
        config, source_type=metadata.source_type, seed=metadata.seed
    )
    return {
        name: np.asarray(values)
        for name, values in simulator.simulate(metadata.num_samples).items()
    }


def simulate_population_mixture(
    components: Sequence[tuple[GraphConfig, int, float]],
    *,
    metadata: PopulationMetadata,
) -> dict[str, NDArray[Any]]:
    """Draw source arrays from a weighted mixture of graph populations.

    Each component is a ``(config, seed, weight)`` triple: the graph, the seed
    its own sampler stream runs on, and its unnormalized mixture weight.
    ``metadata.seed`` seeds only the per-sample component assignment.

    The two seeds are genuinely independent because ``MixtureSimulator`` splits
    ``metadata.seed`` before drawing assignments, so a component seed equal to
    the mixture seed still yields uncorrelated streams. Note also that
    ``MixtureSimulator`` passes each component a derived ``seed`` keyword which
    ``GraphSimulator._simulate_impl`` discards -- a component's draw comes from
    its *construction* seed, which is why one is required per component here.

    Requires at least two components: a single-component mixture would route a
    lone graph through the assignment draw and change its stream for nothing.
    Use :func:`simulate_population` for that case.
    """
    if len(components) < 2:
        raise ValueError(
            "simulate_population_mixture requires at least two components; use "
            "simulate_population for a single graph"
        )

    try:
        from gwmock_pop import MixtureSimulator
    except ImportError as error:
        raise ImportError(
            "Population simulation requires the optional 'gwmock-pop' dependency; "
            "install it with `pip install astrogwb[simulation]`."
        ) from error

    simulators = [
        _graph_simulator(config, source_type=metadata.source_type, seed=seed)
        for config, seed, _ in components
    ]
    simulator = MixtureSimulator(
        simulators,
        [weight for _, _, weight in components],
        seed=metadata.seed,
    )
    return {
        name: np.asarray(values)
        for name, values in simulator.simulate(metadata.num_samples).items()
    }
