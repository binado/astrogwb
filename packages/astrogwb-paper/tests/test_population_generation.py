from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from astrogwb_paper.cli.generate_population import main, simulate_population
from gwmock_pop import GraphSimulator
from gwmock_pop.loaders.file_loader import read_population_catalogue


def _write_graph(path: Path, minimum: float, maximum: float) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "parameters": {
                    "redshift": {
                        "sampler": {
                            "function": "uniform",
                            "arguments": {
                                "minimum": minimum,
                                "maximum": maximum,
                            },
                        }
                    },
                    "marker": {
                        "transform": {
                            "function": "constant_like",
                            "arguments": {
                                "reference": "@redshift",
                                "value": minimum,
                            },
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
    config = _write_graph(tmp_path / "md.yaml", 0.0, 1.0)

    first = simulate_population(config, num_samples=1000, seed=12)
    second = simulate_population(config, num_samples=1000, seed=12)

    np.testing.assert_array_equal(first["redshift"], second["redshift"])


def test_generation_draws_are_a_prefix_stable_stream(tmp_path: Path) -> None:
    """A smaller draw is a bit-identical prefix of a larger one, same seed.

    This pins the property banks and composition prefixes depend on:
    ``GraphSimulator`` draws from its construction-time RNG (``del kwargs`` in
    ``_simulate_impl``), so requesting fewer samples never perturbs the
    stream. A ``gwmock_pop`` upgrade that broke this would silently corrupt
    every bank-prefix / composition-prefix guarantee in
    ``astrogwb_paper.catalogs.compose_catalog`` without touching this file.
    """
    config = _write_graph(tmp_path / "md.yaml", 0.0, 1.0)

    small = simulate_population(config, num_samples=8, seed=5)
    large = simulate_population(config, num_samples=32, seed=5)

    np.testing.assert_array_equal(
        np.asarray(large["redshift"])[:8], np.asarray(small["redshift"])
    )


def test_generate_population_refuses_overwrite_without_force(tmp_path: Path) -> None:
    config = _write_graph(tmp_path / "md.yaml", 0.0, 1.0)
    output = tmp_path / "population.h5"
    arguments = [
        "--config",
        str(config),
        "--num-samples",
        "8",
        "--seed",
        "3",
        "--output",
        str(output),
    ]

    main(arguments)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        main(arguments)

    main([*arguments, "--force"])
    population = read_population_catalogue(output)
    assert len(population["redshift"]) == 8


def test_generate_population_rejects_non_positive_num_samples(tmp_path: Path) -> None:
    config = _write_graph(tmp_path / "md.yaml", 0.0, 1.0)

    with pytest.raises(ValueError, match="num_samples must be > 0"):
        simulate_population(config, num_samples=0, seed=1)


def test_generate_population_no_longer_imports_mixture_simulator() -> None:
    """Mixing moved to compose_catalog; this CLI generates one component."""
    import astrogwb_paper.cli.generate_population as module

    assert module.GraphSimulator is GraphSimulator
    assert not hasattr(module, "MixtureSimulator")
