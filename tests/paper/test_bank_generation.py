"""Population-draw properties that reusable waveform banks rely on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from astrogwb.catalog import PopulationMetadata, simulate_population


def _simple_graph(path: Path, minimum: float, maximum: float) -> Path:
    """A one-parameter graph: enough for the RNG-stream properties."""
    path.write_text(
        yaml.safe_dump(
            {
                "parameters": {
                    "redshift": {
                        "sampler": {
                            "function": "uniform",
                            "arguments": {"minimum": minimum, "maximum": maximum},
                        }
                    },
                    "marker": {
                        "transform": {
                            "function": "constant_like",
                            "arguments": {"reference": "@redshift", "value": minimum},
                        }
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_seeded_generation_is_reproducible(tmp_path: Path) -> None:
    config = _simple_graph(tmp_path / "md.yaml", 0.0, 1.0)
    metadata = PopulationMetadata(
        name="test", seed=12, num_samples=1000, source_type="bns"
    )

    first = simulate_population(config, metadata=metadata)
    second = simulate_population(config, metadata=metadata)

    np.testing.assert_array_equal(first["redshift"], second["redshift"])


def test_generation_draws_are_a_prefix_stable_stream(tmp_path: Path) -> None:
    """A smaller draw is a bit-identical prefix of a larger one, same seed.

    This pins the property banks and composition prefixes depend on:
    ``GraphSimulator`` draws from its construction-time RNG (``del kwargs`` in
    ``_simulate_impl``), so requesting fewer samples never perturbs the
    stream. A ``gwmock_pop`` upgrade that broke this would silently corrupt
    every bank-prefix / composition-prefix guarantee in
    ``CatalogSource.compose`` without touching this file.
    """
    config = _simple_graph(tmp_path / "md.yaml", 0.0, 1.0)
    small_metadata = PopulationMetadata(
        name="test", seed=5, num_samples=8, source_type="bns"
    )
    large_metadata = PopulationMetadata(
        name="test", seed=5, num_samples=32, source_type="bns"
    )

    small = simulate_population(config, metadata=small_metadata)
    large = simulate_population(config, metadata=large_metadata)

    np.testing.assert_array_equal(
        np.asarray(large["redshift"])[:8], np.asarray(small["redshift"])
    )
