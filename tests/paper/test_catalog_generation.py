"""Population-draw properties the persisted catalogs rely on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from astrogwb.catalog import (
    PopulationMetadata,
    simulate_population,
    simulate_population_mixture,
)


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

    This pins what makes variable-catalog-size a clean series: the three
    ``md-imrphenom-s42-n*`` catalogs are separate files drawn independently,
    and they are only nested draws because ``GraphSimulator`` draws from its
    construction-time RNG (``del kwargs`` in ``_simulate_impl``), so requesting
    fewer samples never perturbs the stream. A ``gwmock_pop`` upgrade that
    broke this would silently turn that experiment into three unrelated runs
    without touching this file.
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


def test_a_mixture_draws_each_component_at_its_declared_weight(tmp_path: Path) -> None:
    """Component counts follow the weights, and every sample comes from one of them.

    The two graphs have disjoint support, so which component produced a sample
    is readable off its value alone.
    """
    low = _simple_graph(tmp_path / "low.yaml", 0.0, 1.0)
    high = _simple_graph(tmp_path / "high.yaml", 10.0, 11.0)
    metadata = PopulationMetadata(
        name="test", seed=61, num_samples=4000, source_type="bns"
    )

    drawn = np.asarray(
        simulate_population_mixture(
            [(low, 42, 0.9), (high, 51, 0.1)], metadata=metadata
        )["redshift"]
    )

    from_low = (drawn >= 0.0) & (drawn <= 1.0)
    from_high = (drawn >= 10.0) & (drawn <= 11.0)
    assert np.all(from_low | from_high)
    assert 0.85 < from_low.mean() < 0.95


def test_a_lone_component_does_not_go_through_the_mixture_path(tmp_path: Path) -> None:
    """One component is a plain graph draw, never a one-entry mixture.

    ``MixtureSimulator`` splits its seed to draw component assignments, so
    wrapping a single graph would change its RNG stream -- and with it every
    single-component catalog, forcing all 26 chains to be regenerated.
    ``scripts/generate_catalog.py`` dispatches on component count to avoid
    exactly that; this pins that the two paths really are different.
    """
    config = _simple_graph(tmp_path / "md.yaml", 0.0, 1.0)
    metadata = PopulationMetadata(
        name="test", seed=7, num_samples=64, source_type="bns"
    )

    with pytest.raises(ValueError, match="at least two components"):
        simulate_population_mixture([(config, 7, 1.0)], metadata=metadata)

    # Balanced weights so both components certainly contribute: the mixture
    # path permutes and re-slices the stream, so even over one graph at one
    # seed it cannot reproduce the direct draw.
    direct = np.asarray(simulate_population(config, metadata=metadata)["redshift"])
    mixed = np.asarray(
        simulate_population_mixture(
            [(config, 7, 0.5), (config, 7, 0.5)], metadata=metadata
        )["redshift"]
    )
    assert not np.array_equal(direct, mixed)
