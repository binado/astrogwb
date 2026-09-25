"""Tests for waveform-owned polarization-power generators."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from astrogwb.constants import ISCO_ALPHA
from astrogwb.frequency import uniform_frequency_grid, uniform_grid_spacing
from astrogwb.metadata import WaveformMetadata
from astrogwb.waveform import (
    AnalyticInspiralGenerator,
    RippleGenerator,
)
from astrogwb.waveform.generator._ripple import (
    PRECESSING_MODELS,
    SUPPORTED_APPROXIMANTS,
    TIDAL_MODELS,
    next_smooth_even,
)


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
        WaveformMetadata(
            approximant="TaylorF2",
            sampling_frequency=256.0,
            minimum_frequency=20.0,
            maximum_frequency=100.0,
            reference_frequency=20.0,
            frequency_resolution=4.0,
        )
    )


def test_waveform_metadata_does_not_claim_a_frequency_grid() -> None:
    metadata = WaveformMetadata(
        approximant="TaylorF2",
        minimum_frequency=10.0,
        maximum_frequency=19.0,
        reference_frequency=20.0,
        sampling_frequency=64.0,
        frequency_resolution=2.0,
    )

    assert not hasattr(metadata, "frequencies")
    np.testing.assert_array_equal(
        uniform_frequency_grid(10.0, 19.0, 2.0),
        np.array([10.0, 12.0, 14.0, 16.0, 18.0]),
    )
    assert metadata.frequency_resolution == 2.0


def test_uniform_grid_handles_float_roundoff() -> None:
    metadata = WaveformMetadata(
        approximant="Toy",
        minimum_frequency=0.1,
        maximum_frequency=0.3,
        reference_frequency=0.1,
        sampling_frequency=8.0,
        frequency_resolution=0.1,
    )

    np.testing.assert_allclose(uniform_frequency_grid(0.1, 0.3, 0.1), [0.1, 0.2, 0.3])
    assert metadata.frequency_resolution == 0.1


def test_waveform_metadata_attrs_round_trip_includes_alpha() -> None:
    generator = AnalyticInspiralGenerator(
        WaveformMetadata(
            approximant="AnalyticInspiral",
            minimum_frequency=10.0,
            maximum_frequency=12.0,
            reference_frequency=10.0,
            sampling_frequency=32.0,
            frequency_resolution=2.0,
            alpha=ISCO_ALPHA,
        )
    )

    attrs = generator.metadata.to_attrs()
    assert attrs["alpha"] == ISCO_ALPHA

    restored = WaveformMetadata.from_attrs(attrs, label="toy.h5")
    assert restored.to_attrs() == attrs

    del attrs["reference_frequency"]
    with pytest.raises(ValueError, match="toy.h5: missing waveform metadata"):
        WaveformMetadata.from_attrs(attrs, label="toy.h5")


@pytest.mark.parametrize("use_taper", [True, False])
def test_waveform_metadata_serializes_taper_setting_as_an_integer(
    use_taper: bool,
) -> None:
    metadata = WaveformMetadata(
        approximant="IMRPhenomXAS_NRTidalv3",
        minimum_frequency=10.0,
        maximum_frequency=12.0,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        frequency_resolution=2.0,
        use_taper_in_tidal_corrections=use_taper,
    )

    attrs = metadata.to_attrs()
    assert attrs["use_taper_in_tidal_corrections"] == int(use_taper)
    assert type(attrs["use_taper_in_tidal_corrections"]) is int
    assert WaveformMetadata.from_attrs(attrs, label="toy.h5") == metadata


def test_waveform_metadata_rejects_missing_taper_setting_attribute() -> None:
    metadata = WaveformMetadata(
        approximant="TaylorF2",
        minimum_frequency=10.0,
        maximum_frequency=12.0,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        frequency_resolution=2.0,
    )
    attrs = metadata.to_attrs()
    del attrs["use_taper_in_tidal_corrections"]

    with pytest.raises(ValueError, match="legacy.h5: missing waveform metadata"):
        WaveformMetadata.from_attrs(attrs, label="legacy.h5")


@pytest.mark.parametrize("value", [True, 2, 1.0, "1"])
def test_waveform_metadata_rejects_malformed_taper_setting_attribute(
    value: object,
) -> None:
    attrs: dict[str, Any] = WaveformMetadata(
        approximant="TaylorF2",
        minimum_frequency=10.0,
        maximum_frequency=12.0,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        frequency_resolution=2.0,
    ).to_attrs()
    attrs["use_taper_in_tidal_corrections"] = value

    with pytest.raises(ValueError, match="invalid waveform metadata"):
        WaveformMetadata.from_attrs(attrs, label="broken.h5")


def test_waveform_metadata_rejects_disabled_tidal_taper_for_analytic_inspiral() -> None:
    with pytest.raises(ValidationError, match="use_taper_in_tidal_corrections=False"):
        WaveformMetadata(
            approximant="AnalyticInspiral",
            minimum_frequency=10.0,
            maximum_frequency=12.0,
            reference_frequency=10.0,
            sampling_frequency=32.0,
            frequency_resolution=2.0,
            use_taper_in_tidal_corrections=False,
        )


def test_waveform_metadata_keeps_strict_validation_for_unknown_settings() -> None:
    settings = {
        "approximant": "TaylorF2",
        "minimum_frequency": 10.0,
        "maximum_frequency": 12.0,
        "reference_frequency": 10.0,
        "sampling_frequency": 32.0,
        "frequency_resolution": 2.0,
        "unsupported_setting": 1,
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WaveformMetadata.model_validate(settings)


def test_from_attrs_rejects_a_non_numeric_frequency_attribute() -> None:
    generator = AnalyticInspiralGenerator(
        WaveformMetadata(
            approximant="AnalyticInspiral",
            minimum_frequency=10.0,
            maximum_frequency=12.0,
            reference_frequency=10.0,
            sampling_frequency=32.0,
            frequency_resolution=2.0,
            alpha=ISCO_ALPHA,
        )
    )
    attrs = generator.metadata.to_attrs()
    attrs["minimum_frequency"] = "not-a-number"

    with pytest.raises(ValueError, match="toy.h5: invalid waveform metadata"):
        WaveformMetadata.from_attrs(attrs, label="toy.h5")


def test_from_attrs_rejects_a_non_numeric_alpha_attribute() -> None:
    generator = AnalyticInspiralGenerator(
        WaveformMetadata(
            approximant="AnalyticInspiral",
            minimum_frequency=10.0,
            maximum_frequency=12.0,
            reference_frequency=10.0,
            sampling_frequency=32.0,
            frequency_resolution=2.0,
            alpha=ISCO_ALPHA,
        )
    )
    attrs = generator.metadata.to_attrs()
    attrs["alpha"] = "not-a-number"

    with pytest.raises(ValueError, match="toy.h5: invalid waveform metadata"):
        WaveformMetadata.from_attrs(attrs, label="toy.h5")


def test_waveform_metadata_builds_a_concrete_generator() -> None:
    metadata = WaveformMetadata(
        approximant="TaylorF2",
        minimum_frequency=10.0,
        maximum_frequency=12.0,
        reference_frequency=10.0,
        sampling_frequency=32.0,
        frequency_resolution=2.0,
    )

    generator = metadata.build()
    assert isinstance(generator, RippleGenerator)
    assert generator.metadata is metadata


def test_build_power_kernel_disables_taper_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ripplegw

    waveform = Mock()
    monkeypatch.setattr(ripplegw, "waveform", waveform)

    from astrogwb.waveform.generator._ripple import build_power_kernel

    build_power_kernel("TaylorF2", 20.0)
    waveform.assert_called_once_with("TaylorF2", f_ref=20.0)

    waveform.reset_mock()
    build_power_kernel(
        "IMRPhenomXAS_NRTidalv3",
        20.0,
        use_taper_in_tidal_corrections=False,
    )
    waveform.assert_called_once_with(
        "IMRPhenomXAS_NRTidalv3", f_ref=20.0, no_taper=True
    )


def test_build_power_kernel_rejects_disabled_tidal_taper_for_non_tidal_ripple() -> None:
    from astrogwb.waveform.generator._ripple import build_power_kernel

    with pytest.raises(ValueError, match="only supported for tidal Ripple"):
        build_power_kernel("IMRPhenomD", 20.0, use_taper_in_tidal_corrections=False)


def test_build_power_kernel_rejects_tidal_ripple_without_taper_setting_support() -> (
    None
):
    from astrogwb.waveform.generator._ripple import build_power_kernel

    with pytest.raises(ValueError, match="does not support"):
        build_power_kernel("TaylorF2", 20.0, use_taper_in_tidal_corrections=False)


def test_ripple_generator_calls_its_kernel_with_the_full_grid_above_dc(
    ripple_generator: RippleGenerator,
) -> None:
    """Ripple sees every bin above DC; the band comes from its output.

    Several supported models do not evaluate pointwise in frequency, so a
    generator that handed Ripple only the in-band bins would quietly change
    the in-band values. Mocking the kernel is what pins that direction.

    The DC bin is excluded at the input instead: Ripple evaluates it to NaN,
    and that NaN cannot be masked out of a gradient afterwards. Dropping it
    leaves the spacing Ripple reads off the bottom of the grid intact, which
    is why the exclusion does not perturb the in-band values.
    """
    delta_f = 256.0 / ripple_generator.n_samples
    n_grid = ripple_generator.n_samples // 2
    kernel = Mock(return_value=jnp.ones((n_grid, 2), dtype=jnp.float64))
    object.__setattr__(ripple_generator, "_kernel", kernel)

    power = ripple_generator.generate_batch(_ripple_sources())

    kernel.assert_called_once()
    frequencies, events = kernel.call_args.args
    np.testing.assert_array_equal(
        np.asarray(frequencies), np.arange(1, n_grid + 1) * delta_f
    )
    assert set(events) >= {"M_c", "eta", "d_L", "iota"}
    np.testing.assert_array_equal(
        power, np.ones((ripple_generator.frequencies.size, 2))
    )


def test_ripple_generate_batch_has_no_nan_gradient(
    ripple_generator: RippleGenerator,
) -> None:
    """No NaN reaches reverse mode, which masking the output could not fix.

    With ``f = 0`` in the input grid the DC bin evaluates to NaN, and its
    derivative stays NaN however the forward value is masked or sliced --
    source parameters broadcast across frequency, so their VJP sums every bin
    and ``0 * NaN`` comes back. Excluding the bin is what makes this finite.
    """
    sources = _ripple_sources()

    def total_power(distance: jax.Array) -> jax.Array:
        return ripple_generator.generate_batch(
            {**sources, "luminosity_distance": distance}
        ).sum()

    gradient = jax.grad(total_power)(
        jnp.asarray(sources["luminosity_distance"], dtype=jnp.float64)
    )

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.asarray(gradient) < 0.0)


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


def test_ripple_generate_rejects_multiple_events(
    ripple_generator: RippleGenerator,
) -> None:
    with pytest.raises(ValueError, match="single source"):
        ripple_generator.generate(_ripple_sources())


@pytest.mark.integration
def test_ripple_generate_accepts_zero_dimensional_scalars(
    ripple_generator: RippleGenerator,
) -> None:
    sources = _ripple_sources()
    arrays = {name: values[:1] for name, values in sources.items()}
    scalars = {name: values[0] for name, values in sources.items()}

    np.testing.assert_array_equal(
        ripple_generator.generate(scalars), ripple_generator.generate(arrays)
    )


@pytest.mark.parametrize(
    ("sampling_frequency", "message"),
    [
        (0.0, "sampling_frequency"),
        (np.inf, "sampling_frequency"),
        (np.nan, "sampling_frequency"),
        (1.0e-12, "fewer than two bins"),
    ],
)
def test_ripple_generator_rejects_invalid_sampling_frequency(
    sampling_frequency: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RippleGenerator(
            WaveformMetadata(
                approximant="TaylorF2",
                sampling_frequency=sampling_frequency,
                minimum_frequency=20.0,
                maximum_frequency=100.0,
                reference_frequency=20.0,
                frequency_resolution=4.0,
            )
        )


def test_ripple_generator_rejects_non_aligned_minimum_frequency() -> None:
    """Misalignment is a constructor error, not something a catalog discovers.

    The generator owns Ripple's grid rule, so the effective resolution follows
    from the configuration alone. Catching this at construction is what keeps
    a run from drawing a whole catalog before finding out its band never
    started where it was asked to -- and is why this test no longer needs to
    generate a waveform, so it is no longer ``integration``.
    """
    with pytest.raises(ValueError, match="not on Ripple's frequency grid"):
        RippleGenerator(
            WaveformMetadata(
                approximant="TaylorF2",
                sampling_frequency=256.0,
                minimum_frequency=21.0,
                maximum_frequency=100.0,
                reference_frequency=20.0,
                frequency_resolution=4.0,
            )
        )


@pytest.mark.integration
def test_ripple_generator_infers_df_from_its_own_5_smooth_grid() -> None:
    """The regression this whole change exists for.

    At ``sampling_frequency=1234.0, frequency_resolution=1.0`` astrogwb's old
    power-of-two replica predicted ``n=1234, df=1.0``; Ripple's real
    ``_next_smooth_even`` rounds to ``n=1250``, giving
    ``delta_f=1234/1250=0.9872`` -- 1.28% off. The old code accepted
    ``minimum_frequency=2.0`` silently and stamped the wrong df into the file;
    this generator must instead reject it at construction, and accept only the
    frequency that is actually on Ripple's grid.
    """
    with pytest.raises(ValueError, match="not on Ripple's frequency grid"):
        RippleGenerator(
            WaveformMetadata(
                approximant="TaylorF2",
                sampling_frequency=1234.0,
                minimum_frequency=2.0,
                maximum_frequency=100.0,
                reference_frequency=2.0,
                frequency_resolution=1.0,
            )
        )

    aligned = RippleGenerator(
        WaveformMetadata(
            approximant="TaylorF2",
            sampling_frequency=1234.0,
            minimum_frequency=2.0 * (1234.0 / 1250.0),
            maximum_frequency=100.0,
            reference_frequency=2.0,
            frequency_resolution=1.0,
        )
    )
    frequencies, _ = aligned(_ripple_sources())
    assert uniform_grid_spacing(np.asarray(frequencies)) == pytest.approx(
        0.9872, abs=1e-15
    )


# --------------------------------------------------------------------- #
# The grid rule astrogwb now owns
# --------------------------------------------------------------------- #
def _is_smooth_even(value: int) -> bool:
    """Brute-force oracle: even, and no prime factor above 5."""
    if value <= 0 or value % 2:
        return False
    remainder = value
    for factor in (2, 3, 5):
        while remainder % factor == 0:
            remainder //= factor
    return remainder == 1


@pytest.mark.parametrize("minimum", [1, 2, 3, 7, 100, 309, 1234, 4097, 8192, 8193])
def test_next_smooth_even_is_the_smallest_admissible_length(minimum: int) -> None:
    """Ripple sizes its segment this way, so astrogwb has to agree bin for bin."""
    result = next_smooth_even(minimum)

    assert _is_smooth_even(result)
    assert result >= max(minimum, 2)
    assert not any(
        _is_smooth_even(candidate) for candidate in range(max(minimum, 2), result)
    )


def test_generator_grid_matches_ripples_own_5_smooth_rounding() -> None:
    """``fs=1234`` rounds to 1250, not 1234: the case a power-of-two rule got wrong."""
    generator = RippleGenerator(
        WaveformMetadata(
            approximant="TaylorF2",
            sampling_frequency=1234.0,
            minimum_frequency=2.0 * (1234.0 / 1250.0),
            maximum_frequency=100.0,
            reference_frequency=2.0,
            frequency_resolution=1.0,
        )
    )

    assert generator.n_samples == 1250
    assert generator.segment_duration == 1.0
    assert uniform_grid_spacing(np.asarray(generator.frequencies)) == pytest.approx(
        0.9872, abs=1e-15
    )


def test_ripple_frequencies_are_available_before_generating(
    ripple_generator: RippleGenerator,
) -> None:
    """No warm-up call: the grid follows from the configuration alone.

    This is what lets a reduction over an empty catalog size its accumulator
    without evaluating a waveform.
    """
    frequencies = ripple_generator.frequencies

    assert frequencies.size > 1
    assert float(frequencies[0]) == pytest.approx(20.0, abs=1e-12)
    assert float(frequencies[-1]) <= 100.0


def test_ripple_generator_requires_x64(ripple_generator: RippleGenerator) -> None:
    """Realistic polarization powers underflow in float32."""
    enabled = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="requires JAX x64"):
            ripple_generator.generate_batch(_ripple_sources())
    finally:
        jax.config.update("jax_enable_x64", enabled)


# --------------------------------------------------------------------- #
# Validation is eager, and deliberately absent from the generate path
# --------------------------------------------------------------------- #
def test_check_sources_rejects_values_the_approximant_cannot_carry() -> None:
    """An aligned-spin model would otherwise ignore in-plane spin in silence."""
    generator = RippleGenerator(
        WaveformMetadata(
            approximant="IMRPhenomXAS",
            sampling_frequency=256.0,
            minimum_frequency=20.0,
            maximum_frequency=100.0,
            reference_frequency=20.0,
            frequency_resolution=4.0,
        )
    )

    with pytest.raises(ValueError, match="aligned-spin model"):
        generator.check_sources(_ripple_sources() | {"spin_1x": np.array([0.1, 0.0])})
    with pytest.raises(ValueError, match="no tidal deformability"):
        generator.check_sources(
            _ripple_sources() | {"lambda_1": np.array([300.0, 0.0])}
        )


def test_check_sources_rejects_negative_tidal_deformability(
    ripple_generator: RippleGenerator,
) -> None:
    with pytest.raises(ValueError, match="lambda_1 must be non-negative"):
        ripple_generator.check_sources(
            _ripple_sources() | {"lambda_1": np.array([-1.0, 0.0])}
        )


def test_generate_batch_does_not_validate_values(
    ripple_generator: RippleGenerator,
) -> None:
    """The split this design rests on, pinned rather than assumed.

    Value checks would need a host sync on every chunk, so ``generate_batch``
    must let through exactly what ``check_sources`` rejects -- otherwise the
    generator is not usable inside a traced model.
    """
    sources = _ripple_sources() | {"lambda_1": np.array([-1.0, 0.0])}

    with pytest.raises(ValueError, match="non-negative"):
        ripple_generator.check_sources(sources)

    power = ripple_generator.generate_batch(sources)
    assert power.shape == (ripple_generator.frequencies.size, 2)


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ({"detector_frame_mass_1": np.array([1.4])}, "missing required"),
        (
            _ripple_sources() | {"inclination": np.array([[0.0, 0.4]])},
            "must be 1-D",
        ),
        (
            _ripple_sources() | {"inclination": np.array([0.0, 0.4, 0.2])},
            "expected 2",
        ),
    ],
)
def test_generate_batch_still_rejects_names_and_shapes(
    ripple_generator: RippleGenerator, sources: dict[str, np.ndarray], message: str
) -> None:
    """Names and shapes are static, so they raise under tracing too."""
    with pytest.raises(ValueError, match=message):
        ripple_generator.generate_batch(sources)
    with pytest.raises(ValueError, match=message):
        jax.jit(ripple_generator.generate_batch)(sources)


def test_unsupported_approximant_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unsupported approximant"):
        RippleGenerator(
            WaveformMetadata(
                approximant="NotAWaveform",
                sampling_frequency=256.0,
                minimum_frequency=20.0,
                maximum_frequency=100.0,
                reference_frequency=20.0,
                frequency_resolution=4.0,
            )
        )


# --------------------------------------------------------------------- #
# Parity against gwmock-signal, the GPL backend this adapter replaces
# --------------------------------------------------------------------- #
def _gwmock_reference(
    approximant: str,
    sources: dict[str, np.ndarray],
    *,
    sampling_frequency: float,
    minimum_frequency: float,
    maximum_frequency: float,
    reference_frequency: float,
    segment_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency axis and in-band power from ``gwmock_signal.RippleBackend``."""
    from gwmock_signal.waveform import RippleBackend

    backend = RippleBackend(
        f_ref=reference_frequency, segment_duration=segment_duration
    )
    polarizations = backend.generate_fd_polarizations_batch(
        approximant,
        sampling_frequency=sampling_frequency,
        minimum_frequency=minimum_frequency,
        parameters={name: jnp.asarray(v) for name, v in sources.items()},
    )
    mask = (polarizations.frequencies >= minimum_frequency) & (
        polarizations.frequencies <= maximum_frequency
    )
    power = (
        jnp.abs(polarizations.plus[:, mask]) ** 2
        + jnp.abs(polarizations.cross[:, mask]) ** 2
    ).T
    return np.asarray(polarizations.frequencies[mask]), np.asarray(power)


#: Sources spanning every degree of freedom the supported families take.
def _parity_sources(approximant: str) -> dict[str, np.ndarray]:
    sources = {
        "detector_frame_mass_1": np.array([1.6, 2.0]),
        "detector_frame_mass_2": np.array([1.3, 1.5]),
        "luminosity_distance": np.array([100.0, 220.0]),
        "inclination": np.array([0.3, 1.1]),
        "coa_phase": np.array([0.2, 1.9]),
        "spin_1z": np.array([0.05, -0.01]),
        "spin_2z": np.array([-0.02, 0.03]),
    }
    if approximant in TIDAL_MODELS:
        sources |= {
            "lambda_1": np.array([300.0, 120.0]),
            "lambda_2": np.array([400.0, 210.0]),
        }
    if approximant in PRECESSING_MODELS:
        sources |= {
            "detector_frame_mass_1": np.array([32.0, 28.0]),
            "detector_frame_mass_2": np.array([25.0, 20.0]),
            "spin_1x": np.array([0.1, -0.05]),
            "spin_1y": np.array([0.05, 0.02]),
            "spin_2x": np.array([-0.05, 0.03]),
            "spin_2y": np.array([0.02, -0.04]),
        }
    return sources


@pytest.mark.integration
@pytest.mark.parametrize("approximant", SUPPORTED_APPROXIMANTS)
def test_power_matches_the_gwmock_backend(approximant: str) -> None:
    """Every supported family, against the GPL backend this replaces.

    Exact on the frequency axis: the grid rule is reimplemented, so any
    disagreement there is a bug, not rounding.

    Approximate on power, with ``atol=0`` so nothing hides near ``1e-47``. The
    residual is XLA fusion, not the dropped cutoff window -- this adapter's own
    kernel differs from itself between an eager and a jitted call by the same
    order, and the window is separately shown to be the identity in band.
    ``2e-13`` was the largest deviation measured across these families (on
    ``IMRPhenomHM``: the multi-mode models accumulate more of it than the
    single-mode ones), so ``1e-11`` leaves about two decades of headroom while
    still pinning agreement to eleven significant figures.

    The families come from Ripple's registry, gwmock's from a list of its own,
    so a Ripple release can add one gwmock cannot yet generate (ripplegw 0.4.0
    added ``IMRPhenomXP_NRTidalv3``). There is no reference to compare such a
    family against, so it is skipped rather than failed -- and asked of
    gwmock's public list rather than named, so it rejoins once gwmock has it.
    """
    from gwmock_signal.waveform import RippleBackend

    if approximant not in RippleBackend().available_approximants():
        pytest.skip(f"gwmock-signal has no {approximant} to compare against")
    settings = {
        "sampling_frequency": 512.0,
        "minimum_frequency": 16.0,
        "maximum_frequency": 64.0,
        "reference_frequency": 16.0,
    }
    sources = _parity_sources(approximant)
    generator = RippleGenerator(
        WaveformMetadata.model_validate(
            {"approximant": approximant, "frequency_resolution": 1.0, **settings}
        )
    )
    frequencies, power = _gwmock_reference(
        approximant, sources, segment_duration=1.0, **settings
    )

    # Jitted, like every real caller: the generator carries no jit of its own,
    # and the precessing models are far too slow to evaluate op by op.
    ours = jax.jit(generator.generate_batch)(sources)

    np.testing.assert_array_equal(np.asarray(generator.frequencies), frequencies)
    np.testing.assert_allclose(np.asarray(ours), power, rtol=1e-11, atol=0.0)


@pytest.mark.integration
def test_dropping_the_cutoff_window_changes_nothing_in_band() -> None:
    """The reason this adapter carries no window at all.

    gwmock tapers below ``minimum_frequency`` because it inverse-transforms to
    the time domain and a hard truncation rings across the buffer. astrogwb
    never leaves the frequency domain, and the taper reaches exactly 1 at
    ``minimum_frequency`` -- so on every bin astrogwb keeps, the window is the
    identity. Asserted here rather than argued in a comment.
    """
    from gwmock_signal.waveform.backends.ripple import _cutoff_window

    generator = RippleGenerator(
        WaveformMetadata(
            approximant="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=512.0,
            minimum_frequency=16.0,
            maximum_frequency=64.0,
            reference_frequency=16.0,
            frequency_resolution=1.0,
        )
    )
    full_grid = jnp.arange(generator.n_samples // 2 + 1) * (512.0 / generator.n_samples)
    window = jnp.asarray(_cutoff_window(full_grid, 16.0, 0.05, jnp))

    in_band = full_grid >= 16.0
    np.testing.assert_array_equal(
        np.asarray(window[in_band]), np.ones(int(in_band.sum()))
    )


@pytest.mark.integration
def test_untapered_nrtidal_power_survives_the_normal_merger_cutoff() -> None:
    sources = {
        "detector_frame_mass_1": np.array([1.4]),
        "detector_frame_mass_2": np.array([1.3]),
        "inclination": np.array([0.4]),
        "luminosity_distance": np.array([100.0]),
        "lambda_1": np.array([400.0]),
        "lambda_2": np.array([500.0]),
    }
    tapered = RippleGenerator(
        WaveformMetadata(
            approximant="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=8192.0,
            minimum_frequency=2.0,
            maximum_frequency=4096.0,
            reference_frequency=20.0,
            frequency_resolution=1.0,
        )
    )
    untapered = RippleGenerator(
        WaveformMetadata(
            approximant="IMRPhenomXAS_NRTidalv3",
            sampling_frequency=8192.0,
            minimum_frequency=2.0,
            maximum_frequency=4096.0,
            reference_frequency=20.0,
            frequency_resolution=1.0,
            use_taper_in_tidal_corrections=False,
        )
    )

    frequencies, tapered_power = tapered(sources)
    _, untapered_power = untapered(sources)
    high_frequency = np.asarray(frequencies) > 3000.0

    assert np.all(np.asarray(tapered_power)[high_frequency] == 0.0)
    assert np.all(np.asarray(untapered_power)[high_frequency] > 0.0)


@pytest.mark.integration
def test_generate_batch_is_traceable_and_values_do_not_recompile(
    ripple_generator: RippleGenerator,
) -> None:
    """The whole point: a Ripple catalog can be drawn inside a jitted model."""
    sources = _ripple_sources()
    traces = 0

    def counted(catalog):
        # The body runs once per trace, so this counts compilations.
        nonlocal traces
        traces += 1
        return ripple_generator.generate_batch(catalog)

    jitted = jax.jit(counted)

    np.testing.assert_allclose(
        np.asarray(jitted(sources)),
        np.asarray(ripple_generator.generate_batch(sources)),
        rtol=1e-13,
        atol=0.0,
    )

    shifted = {name: values * 1.01 for name, values in sources.items()}
    assert np.all(np.asarray(jitted(shifted)) > 0.0)
    assert traces == 1, "changing parameter values must not recompile"


@pytest.mark.integration
def test_generate_is_the_length_one_batch(ripple_generator: RippleGenerator) -> None:
    """``generate`` is the inherited default now that the kernel is vmapped."""
    sources = _ripple_sources()
    single = {name: values[:1] for name, values in sources.items()}

    np.testing.assert_allclose(
        np.asarray(ripple_generator.generate(single)),
        np.asarray(ripple_generator.generate_batch(sources))[:, 0],
        rtol=1e-13,
        atol=0.0,
    )


@pytest.mark.integration
def test_polarization_power_is_float64(ripple_generator: RippleGenerator) -> None:
    assert ripple_generator.generate_batch(_ripple_sources()).dtype == jnp.float64
