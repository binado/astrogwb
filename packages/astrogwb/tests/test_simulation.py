"""Tests for population drawing and waveform-catalog generation adapters."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from astrogwb.constants import ISCO_ALPHA
from astrogwb.simulation import (
    AnalyticInspiralGenerator,
    GeneratedPolarizationPower,
    generate_catalog,
    simulate_population,
)
from astrogwb.waveform import inspiral_polarization_power
from numpy.typing import ArrayLike


@pytest.fixture
def source_parameters() -> dict[str, np.ndarray]:
    return {
        "source_frame_mass_1": np.array([1.4, 1.2]),
        "source_frame_mass_2": np.array([1.3, 1.1]),
        "redshift": np.array([0.1, 0.2]),
        "luminosity_distance": np.array([400.0, 900.0]),
        "inclination": np.array([0.0, 1.0]),
        "caller_owned": np.array([7.0, 8.0]),
    }


def test_fake_generator_flows_through_catalog_with_metadata_and_parameters(
    source_parameters: dict[str, np.ndarray],
) -> None:
    expected_power = np.array([[1.0, 2.0], [3.0, 4.0]])

    class FakeGenerator:
        received: Mapping[str, ArrayLike] | None = None

        def __call__(
            self, source_parameters: Mapping[str, ArrayLike]
        ) -> GeneratedPolarizationPower:
            self.received = source_parameters
            return GeneratedPolarizationPower(
                frequencies=np.array([10.0, 12.0]),
                polarization_power=expected_power,
                approximant="FakeWaveform",
                minimum_frequency=10.0,
                maximum_frequency=12.5,
                reference_frequency=11.0,
                sampling_frequency=64.0,
                df=2.0,
            )

    generator = FakeGenerator()
    catalog = generate_catalog(
        source_parameters,
        generator=generator,
        extra_attrs={"producer": "test", "seed": 42},
    )

    assert generator.received is source_parameters
    np.testing.assert_array_equal(catalog.polarization_power.values, expected_power)
    assert catalog.parameter.values.tolist() == list(source_parameters)
    for name, values in source_parameters.items():
        np.testing.assert_array_equal(
            catalog.source_parameters.sel(parameter=name), values
        )
    assert catalog.attrs == {
        "format_name": "waveform_catalog",
        "domain": "frequency",
        "approximant": "FakeWaveform",
        "minimum_frequency": 10.0,
        "maximum_frequency": 12.5,
        "reference_frequency": 11.0,
        "sampling_frequency": 64.0,
        "df": 2.0,
        "producer": "test",
        "seed": 42,
    }


@pytest.mark.parametrize(
    ("maximum_frequency", "expected"),
    [
        (20.0, np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])),
        (19.0, np.array([10.0, 12.0, 14.0, 16.0, 18.0])),
    ],
)
def test_analytic_generator_builds_grid_through_largest_in_band_bin(
    source_parameters: dict[str, np.ndarray],
    maximum_frequency: float,
    expected: np.ndarray,
) -> None:
    generated = AnalyticInspiralGenerator(
        minimum_frequency=10.0,
        maximum_frequency=maximum_frequency,
        df=2.0,
        alpha=ISCO_ALPHA,
    )(source_parameters)

    np.testing.assert_array_equal(generated.frequencies, expected)
    assert generated.maximum_frequency == maximum_frequency


def test_analytic_generator_matches_direct_power_calculation(
    source_parameters: dict[str, np.ndarray],
) -> None:
    generator = AnalyticInspiralGenerator(
        minimum_frequency=10.0,
        maximum_frequency=15.0,
        df=2.0,
        alpha=ISCO_ALPHA,
    )

    generated = generator(source_parameters)
    expected = np.asarray(
        inspiral_polarization_power(
            generated.frequencies, source_parameters, alpha=ISCO_ALPHA
        )
    ).T

    np.testing.assert_array_equal(generated.polarization_power, expected)
    assert generated.approximant == "AnalyticInspiral"
    assert generated.reference_frequency == 10.0
    assert generated.sampling_frequency == 30.0
    assert generated.df == 2.0


@pytest.mark.parametrize(
    ("minimum_frequency", "maximum_frequency", "df"),
    [
        (np.nan, 20.0, 1.0),
        (10.0, np.inf, 1.0),
        (10.0, 20.0, np.nan),
        (10.0, 20.0, 0.0),
        (10.0, 20.0, -1.0),
        (10.0, 10.0, 1.0),
        (20.0, 10.0, 1.0),
    ],
)
def test_analytic_generator_rejects_invalid_grid_settings(
    source_parameters: dict[str, np.ndarray],
    minimum_frequency: float,
    maximum_frequency: float,
    df: float,
) -> None:
    generator = AnalyticInspiralGenerator(
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        df=df,
        alpha=ISCO_ALPHA,
    )

    with pytest.raises(ValueError):
        generator(source_parameters)


def test_generate_catalog_rejects_malformed_generator_output(
    source_parameters: dict[str, np.ndarray],
) -> None:
    def malformed_generator(
        source_parameters: Mapping[str, ArrayLike],
    ) -> GeneratedPolarizationPower:
        del source_parameters
        return GeneratedPolarizationPower(
            frequencies=np.array([10.0, 12.0]),
            polarization_power=np.ones((3, 2)),
            approximant="Malformed",
            minimum_frequency=10.0,
            maximum_frequency=12.0,
            reference_frequency=10.0,
            sampling_frequency=24.0,
            df=2.0,
        )

    with pytest.raises(ValueError, match="conflicting sizes"):
        generate_catalog(source_parameters, generator=malformed_generator)


def test_simulate_population_accepts_mapping_and_path_and_is_prefix_stable(
    tmp_path: Path,
) -> None:
    graph = {
        "draw": {
            "sampler": {
                "function": "uniform",
                "arguments": {"minimum": 0.0, "maximum": 1.0},
            }
        }
    }
    config_path = tmp_path / "population.yaml"
    config_path.write_text(
        """\
name: test-population
parameters:
  draw:
    sampler:
      function: uniform
      arguments:
        minimum: 0.0
        maximum: 1.0
""",
        encoding="utf-8",
    )

    mapping_draw = simulate_population(
        graph, num_samples=4, seed=123, source_type="bns"
    )
    repeated_draw = simulate_population(
        graph, num_samples=4, seed=123, source_type="bns"
    )
    longer_draw = simulate_population(graph, num_samples=8, seed=123, source_type="bns")
    path_draw = simulate_population(
        config_path, num_samples=4, seed=123, source_type="bns"
    )
    mapping_values = np.asarray(mapping_draw["draw"])
    repeated_values = np.asarray(repeated_draw["draw"])
    longer_values = np.asarray(longer_draw["draw"])
    path_values = np.asarray(path_draw["draw"])

    assert type(mapping_draw) is dict
    np.testing.assert_array_equal(mapping_values, repeated_values)
    np.testing.assert_array_equal(mapping_values, longer_values[:4])
    np.testing.assert_array_equal(mapping_values, path_values)
    assert mapping_values.dtype == repeated_values.dtype


def test_population_simulation_missing_extra_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "gwmock_pop", None)

    with pytest.raises(ImportError, match=r"pip install astrogwb\[simulation\]"):
        simulate_population({}, num_samples=1, seed=0)
