from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from astrogwb_paper.cli.generate_population import main, simulate_population
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


def test_seeded_mixture_is_reproducible_and_uses_both_components(
    tmp_path: Path,
) -> None:
    md = _write_graph(tmp_path / "md.yaml", 0.0, 1.0)
    uniform = _write_graph(tmp_path / "uniform.yaml", 10.0, 11.0)

    first = simulate_population(
        md,
        uniform,
        uniform_mixing_fraction=0.2,
        num_samples=1000,
        seed=12,
    )
    second = simulate_population(
        md,
        uniform,
        uniform_mixing_fraction=0.2,
        num_samples=1000,
        seed=12,
    )

    np.testing.assert_array_equal(first["redshift"], second["redshift"])
    assert np.mean(np.asarray(first["marker"]) == 10.0) == pytest.approx(0.2, abs=0.04)


@pytest.mark.parametrize(
    ("epsilon", "expected_marker"),
    [(0.0, 0.0), (1.0, 10.0)],
)
def test_endpoint_fraction_uses_only_selected_graph(
    tmp_path: Path, epsilon: float, expected_marker: float
) -> None:
    md = _write_graph(tmp_path / "md.yaml", 0.0, 1.0)
    uniform = _write_graph(tmp_path / "uniform.yaml", 10.0, 11.0)

    population = simulate_population(
        md,
        uniform,
        uniform_mixing_fraction=epsilon,
        num_samples=16,
        seed=4,
    )

    np.testing.assert_array_equal(population["marker"], expected_marker)


def test_generate_population_refuses_overwrite_without_force(tmp_path: Path) -> None:
    md = _write_graph(tmp_path / "md.yaml", 0.0, 1.0)
    uniform = _write_graph(tmp_path / "uniform.yaml", 10.0, 11.0)
    output = tmp_path / "population.h5"
    arguments = [
        "--md-config",
        str(md),
        "--uniform-redshift-config",
        str(uniform),
        "--uniform-mixing-fraction",
        "0.2",
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
    assert "proposal_component" not in population
    assert "proposal_redshift_logpdf" not in population


@pytest.mark.parametrize("epsilon", [-0.1, 1.1])
def test_generate_population_rejects_invalid_fraction(
    tmp_path: Path, epsilon: float
) -> None:
    with pytest.raises(ValueError, match="0 <= epsilon <= 1"):
        simulate_population(
            tmp_path / "md.yaml",
            tmp_path / "uniform.yaml",
            uniform_mixing_fraction=epsilon,
            num_samples=1,
            seed=1,
        )
