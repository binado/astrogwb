"""Tests for waveform-owned polarization-power generators."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from astrogwb.frequency import uniform_frequency_grid, uniform_grid_spacing
from astrogwb.waveform import PolarizationPowerGenerator, RippleGenerator


def _ripple_sources() -> dict[str, np.ndarray]:
    return {
        "detector_frame_mass_1": np.array([1.4, 1.3]),
        "detector_frame_mass_2": np.array([1.3, 1.2]),
        "inclination": np.array([0.0, 0.4]),
        "luminosity_distance": np.array([100.0, 200.0]),
    }


@pytest.fixture
def ripple_generator() -> RippleGenerator:
    return RippleGenerator(
        approximant="TaylorF2",
        sampling_frequency=256.0,
        minimum_frequency=20.0,
        maximum_frequency=100.0,
        reference_frequency=20.0,
        frequency_resolution=4.0,
        chunk_size=1,
    )


def test_base_generator_is_metadata_only_without_a_frequency_grid() -> None:
    generator = PolarizationPowerGenerator(
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=19.0,
        reference_frequency=20.0,
        sampling_frequency=64.0,
        frequency_resolution=2.0,
    )

    assert not hasattr(generator, "frequencies")
    np.testing.assert_array_equal(
        uniform_frequency_grid(10.0, 19.0, 2.0),
        np.array([10.0, 12.0, 14.0, 16.0, 18.0]),
    )
    assert generator.frequency_resolution == 2.0


def test_uniform_grid_handles_float_roundoff() -> None:
    generator = PolarizationPowerGenerator(
        approximant="Toy",
        minimum_frequency=0.1,
        maximum_frequency=0.3,
        reference_frequency=0.1,
        sampling_frequency=8.0,
        frequency_resolution=0.1,
    )

    np.testing.assert_allclose(uniform_frequency_grid(0.1, 0.3, 0.1), [0.1, 0.2, 0.3])
    assert generator.frequency_resolution == 0.1


def test_base_generator_is_a_metadata_only_descriptor() -> None:
    generator = PolarizationPowerGenerator(
        approximant="Toy",
        minimum_frequency=10.0,
        maximum_frequency=12.0,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        frequency_resolution=2.0,
    )

    with pytest.raises(NotImplementedError, match="metadata-only"):
        generator({"detector_frame_mass_1": np.array([1.4])})


@pytest.mark.integration
def test_ripple_generator_owns_grid_and_reduces_chunked_power(
    ripple_generator: RippleGenerator,
) -> None:
    generator = ripple_generator
    assert not hasattr(generator, "frequencies")

    frequencies, power = generator(_ripple_sources())

    assert frequencies[0] == 20.0
    assert frequencies[-1] == 100.0
    # An assertion on the produced grid, not a prediction: this is now a real
    # test of the inference against Ripple's own axis.
    assert uniform_grid_spacing(np.asarray(frequencies)) == 4.0
    assert isinstance(power, jax.Array)
    assert power.shape == (frequencies.size, 2)
    assert power.dtype == np.float64
    assert np.all(power >= 0.0)


def test_ripple_generator_has_no_fabricated_spacing(
    ripple_generator: RippleGenerator,
) -> None:
    """Nothing stands in for the measured grid spacing any more."""
    assert not hasattr(ripple_generator, "df")


@pytest.mark.integration
def test_ripple_generate_matches_the_first_batch_column(
    ripple_generator: RippleGenerator,
) -> None:
    sources = _ripple_sources()
    batch = ripple_generator.generate_batch(sources)
    first = {name: values[:1] for name, values in sources.items()}
    np.testing.assert_allclose(
        ripple_generator.generate(first), batch[:, 0], rtol=1e-12
    )


def test_ripple_generator_rejects_mismatched_source_parameter_shapes(
    ripple_generator: RippleGenerator,
) -> None:
    sources = _ripple_sources()
    sources["inclination"] = np.array([0.0])

    with pytest.raises(ValueError, match="matching shapes"):
        ripple_generator(sources)


def test_ripple_generator_rejects_non_one_dimensional_source_parameters(
    ripple_generator: RippleGenerator,
) -> None:
    sources = {name: np.ones((2, 1)) for name in _ripple_sources()}

    with pytest.raises(ValueError, match="one-dimensional"):
        ripple_generator(sources)


def test_ripple_generator_rejects_empty_source_parameters(
    ripple_generator: RippleGenerator,
) -> None:
    with pytest.raises(ValueError, match="at least one array"):
        ripple_generator({})


def test_ripple_generator_rejects_zero_events(
    ripple_generator: RippleGenerator,
) -> None:
    sources = {name: np.array([]) for name in _ripple_sources()}

    with pytest.raises(ValueError, match="at least one event"):
        ripple_generator(sources)


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


@pytest.mark.integration
def test_ripple_generator_rejects_non_aligned_minimum_frequency() -> None:
    """Misalignment is now detected against the grid Ripple actually built.

    Construction no longer knows Ripple's effective resolution, so this can
    only be caught by generating -- which is what promotes this test to
    ``integration``. The check it becomes is strictly stronger: "did the grid
    Ripple actually built start at f_min?" rather than "is f_min a multiple of
    a number astrogwb guessed?"
    """
    generator = RippleGenerator(
        approximant="TaylorF2",
        sampling_frequency=256.0,
        minimum_frequency=21.0,
        maximum_frequency=100.0,
        reference_frequency=20.0,
        frequency_resolution=4.0,
        chunk_size=1,
    )
    with pytest.raises(ValueError, match="not on Ripple's frequency grid"):
        generator(_ripple_sources())


@pytest.mark.integration
def test_ripple_generator_infers_df_from_its_own_5_smooth_grid() -> None:
    """The regression this whole change exists for.

    At ``sampling_frequency=1234.0, frequency_resolution=1.0`` astrogwb's old
    power-of-two replica predicted ``n=1234, df=1.0``; Ripple's real
    ``_next_smooth_even`` rounds to ``n=1250``, giving
    ``delta_f=1234/1250=0.9872`` -- 1.28% off. The old code accepted
    ``minimum_frequency=2.0`` silently and stamped the wrong df into the file;
    this generator must instead reject it, and accept only the frequency that
    is actually on Ripple's grid.
    """
    misaligned = RippleGenerator(
        approximant="TaylorF2",
        sampling_frequency=1234.0,
        minimum_frequency=2.0,
        maximum_frequency=100.0,
        reference_frequency=2.0,
        frequency_resolution=1.0,
        chunk_size=1,
    )
    with pytest.raises(ValueError, match="not on Ripple's frequency grid"):
        misaligned(_ripple_sources())

    aligned = RippleGenerator(
        approximant="TaylorF2",
        sampling_frequency=1234.0,
        minimum_frequency=2.0 * (1234.0 / 1250.0),
        maximum_frequency=100.0,
        reference_frequency=2.0,
        frequency_resolution=1.0,
        chunk_size=1,
    )
    frequencies, _ = aligned(_ripple_sources())
    assert uniform_grid_spacing(np.asarray(frequencies)) == pytest.approx(
        0.9872, abs=1e-15
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

    frequencies_one_per_chunk, power_one_per_chunk = one_per_chunk(sources)
    frequencies_one_chunk, power_one_chunk = one_chunk(sources)

    np.testing.assert_array_equal(frequencies_one_per_chunk, frequencies_one_chunk)
    np.testing.assert_allclose(power_one_per_chunk, power_one_chunk, rtol=1e-12)
