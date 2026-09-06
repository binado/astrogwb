"""Tests for waveform-owned polarization-power generators."""

from __future__ import annotations

import numpy as np
import pytest

from astrogwb.waveform import PolarizationPowerGenerator, RippleGenerator


def _ripple_sources() -> dict[str, np.ndarray]:
    return {
        "detector_frame_mass_1": np.array([1.4, 1.3]),
        "detector_frame_mass_2": np.array([1.3, 1.2]),
        "inclination": np.array([0.0, 0.4]),
        "luminosity_distance": np.array([100.0, 200.0]),
    }


def test_base_generator_from_bounds_builds_the_owned_grid() -> None:
    generator = PolarizationPowerGenerator.from_bounds(
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=19.0,
        reference_frequency=20.0,
        sampling_frequency=64.0,
        df=2.0,
    )

    np.testing.assert_array_equal(
        generator.frequencies, np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    )
    assert generator.df == 2.0


def test_base_generator_derives_inclusive_grid_with_float_roundoff() -> None:
    generator = PolarizationPowerGenerator.from_bounds(
        approximant="Toy",
        minimum_frequency=0.1,
        maximum_frequency=0.3,
        reference_frequency=0.1,
        sampling_frequency=8.0,
        df=0.1,
    )

    np.testing.assert_allclose(generator.frequencies, [0.1, 0.2, 0.3])
    assert generator.frequencies is not generator.frequencies


def test_base_generator_is_a_metadata_only_descriptor() -> None:
    generator = PolarizationPowerGenerator.from_bounds(
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=12.0,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        df=2.0,
    )

    with pytest.raises(NotImplementedError, match="metadata-only"):
        generator({"detector_frame_mass_1": np.array([1.4])})


@pytest.mark.integration
def test_ripple_generator_owns_grid_and_reduces_chunked_power() -> None:
    generator = RippleGenerator(
        approximant="TaylorF2",
        sampling_frequency=256.0,
        minimum_frequency=20.0,
        maximum_frequency=100.0,
        reference_frequency=20.0,
        frequency_resolution=4.0,
        chunk_size=1,
    )

    power = generator(_ripple_sources())

    assert generator.frequencies[0] == 20.0
    assert generator.frequencies[-1] == 100.0
    assert generator.df == 4.0
    assert power.shape == (generator.frequencies.size, 2)
    assert power.dtype == np.float64
    assert np.all(power >= 0.0)


@pytest.mark.parametrize(
    ("sampling_frequency", "message"),
    [
        (0.0, "finite and positive"),
        (np.inf, "finite and positive"),
        (np.nan, "finite and positive"),
        (1.0e-12, "no Ripple samples"),
    ],
)
def test_ripple_generator_rejects_invalid_sampling_frequency(
    sampling_frequency: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RippleGenerator(
            approximant="TaylorF2",
            sampling_frequency=sampling_frequency,
            minimum_frequency=20.0,
            maximum_frequency=100.0,
            reference_frequency=20.0,
            frequency_resolution=4.0,
            chunk_size=1,
        )


def test_ripple_generator_rejects_non_aligned_minimum_frequency() -> None:
    with pytest.raises(ValueError, match="minimum_frequency must align"):
        RippleGenerator(
            approximant="TaylorF2",
            sampling_frequency=256.0,
            minimum_frequency=21.0,
            maximum_frequency=100.0,
            reference_frequency=20.0,
            frequency_resolution=4.0,
            chunk_size=1,
        )


@pytest.mark.integration
def test_ripple_generator_chunking_preserves_power() -> None:
    sources = _ripple_sources()
    one_per_chunk = RippleGenerator(
        approximant="TaylorF2",
        sampling_frequency=256.0,
        minimum_frequency=20.0,
        maximum_frequency=100.0,
        reference_frequency=20.0,
        frequency_resolution=4.0,
        chunk_size=1,
    )
    one_chunk = RippleGenerator(
        approximant="TaylorF2",
        sampling_frequency=256.0,
        minimum_frequency=20.0,
        maximum_frequency=100.0,
        reference_frequency=20.0,
        frequency_resolution=4.0,
        chunk_size=2,
    )

    np.testing.assert_allclose(one_per_chunk(sources), one_chunk(sources), rtol=1e-12)
