"""Bank generation: the merged population-draw + waveform CLI.

Replaces ``test_population_generation.py``. The population no longer lands on
disk, so what is checked is the draw's determinism (unchanged) and the bank the
CLI writes end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from astrogwb.catalog import PopulationMetadata, simulate_population
from astrogwb.paper.cli.generate_bank import main
from astrogwb.paper.config.banks import read_bank_provenance

BNS_GRAPH = {
    "parameters": {
        "redshift": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": 0.1, "maximum": 1.0},
            }
        },
        "luminosity_distance": {
            "transform": {
                "function": "redshift_to_luminosity_distance",
                "arguments": {
                    "redshift": "@redshift",
                    "hubble_constant": 67.66,
                    "omega_m": 0.3096,
                    "max_redshift": 20.0,
                },
            }
        },
        "source_frame_mass_1": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": 1.2, "maximum": 1.6},
            }
        },
        "source_frame_mass_2": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": 1.2, "maximum": 1.6},
            }
        },
        "spin_1z": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": -0.01, "maximum": 0.01},
            }
        },
        "spin_2z": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": -0.01, "maximum": 0.01},
            }
        },
        "lambda_1": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": 0.0, "maximum": 100.0},
            }
        },
        "lambda_2": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": 0.0, "maximum": 100.0},
            }
        },
        "inclination": {
            "transform": {
                "function": "constant_like",
                "arguments": {"reference": "@redshift", "value": 0.0},
            }
        },
        "coa_phase": {
            "transform": {
                "function": "constant_like",
                "arguments": {"reference": "@redshift", "value": 0.0},
            }
        },
        "coa_time": {
            "transform": {
                "function": "constant_like",
                "arguments": {"reference": "@redshift", "value": 0.0},
            }
        },
    }
}

BANK_TOML = """\
population = "{population}"
seed = 7
num_samples = {num_samples}

[waveform]
approximant = "TaylorF2"
sampling_frequency = 512.0
minimum_frequency = 20.0
maximum_frequency = 128.0
reference_frequency = 20.0
frequency_resolution = 1.0
chunk_size = 2
"""


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


def _workspace(tmp_path: Path, *, num_samples: int = 4) -> Path:
    """A miniature config/ tree the bank CLI can resolve a population from."""
    (tmp_path / "config/populations").mkdir(parents=True)
    (tmp_path / "config/banks").mkdir(parents=True)
    (tmp_path / "config/populations/toy.yaml").write_text(
        yaml.safe_dump(BNS_GRAPH, sort_keys=False), encoding="utf-8"
    )
    config = tmp_path / "config/banks/toy-bank.toml"
    config.write_text(
        BANK_TOML.format(population="toy", num_samples=num_samples), encoding="utf-8"
    )
    return config


# --------------------------------------------------------------------------- #
# The population draw
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# The CLI, end to end
# --------------------------------------------------------------------------- #
def test_a_non_positive_sample_count_is_rejected_by_the_bank_config(
    tmp_path: Path,
) -> None:
    config = _workspace(tmp_path, num_samples=0)

    with pytest.raises(ValueError, match="num_samples"):
        main(["--config", str(config), "--output", str(tmp_path / "bank.h5")])


@pytest.mark.integration
def test_generated_bank_records_its_own_provenance(tmp_path: Path) -> None:
    config = _workspace(tmp_path)
    output = tmp_path / "bank.h5"

    main(["--config", str(config), "--output", str(output)])

    provenance = read_bank_provenance(output)
    assert provenance.population == "toy"
    assert provenance.seed == 7
    assert provenance.num_samples == 4
    assert provenance.redshift_proposal.kind == "uniform_redshift"
    assert provenance.support == (0.1, 1.0)


@pytest.mark.integration
def test_generate_bank_refuses_overwrite_without_force(tmp_path: Path) -> None:
    config = _workspace(tmp_path)
    output = tmp_path / "bank.h5"
    arguments = ["--config", str(config), "--output", str(output)]

    main(arguments)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        main(arguments)

    main([*arguments, "--force"])
    assert read_bank_provenance(output).num_samples == 4
