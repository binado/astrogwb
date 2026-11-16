"""Tests for waveform-owned intensity generators."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from astrogwb.constants import ISCO_ALPHA
from astrogwb.waveform import (
    AnalyticInspiralGenerator,
    RippleGenerator,
    WaveformMetadata,
)
from astrogwb.waveform._ripple import (
    PRECESSING_MODELS,
    SUPPORTED_APPROXIMANTS,
    TIDAL_MODELS,
    next_smooth_even,
    ripple_parameters,
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


def test_waveform_metadata_json_round_trip_includes_alpha() -> None:
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

    attrs = generator.metadata.model_dump(mode="json")
    assert attrs["alpha"] == ISCO_ALPHA

    restored = WaveformMetadata.model_validate_json(json.dumps(attrs))
    assert restored.model_dump(mode="json") == attrs

    del attrs["reference_frequency"]
    with pytest.raises(ValueError, match="reference_frequency"):
        WaveformMetadata.model_validate_json(json.dumps(attrs))


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


def _grid_settings(**overrides: Any) -> dict[str, Any]:
    return {
        "approximant": "TaylorF2",
        "minimum_frequency": 10.0,
        "maximum_frequency": 100.0,
        "reference_frequency": 10.0,
        "sampling_frequency": 256.0,
        **overrides,
    }


def test_waveform_metadata_reads_a_record_without_grid_settings_as_uniform() -> None:
    attrs = WaveformMetadata.model_validate(
        _grid_settings(frequency_resolution=2.0)
    ).model_dump(mode="json")
    del attrs["frequency_spacing"]

    restored = WaveformMetadata.model_validate_json(json.dumps(attrs))

    assert restored.frequency_spacing == "linear"


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"frequency_spacing": "log", "minimum_frequency": 0.0},
            "minimum_frequency > 0",
        ),
    ],
)
def test_waveform_metadata_rejects_inconsistent_grid_settings(
    settings: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        WaveformMetadata.model_validate(_grid_settings(**settings))


def test_ripple_generator_calls_its_kernel_with_the_full_grid_above_dc(
    ripple_generator: RippleGenerator,
) -> None:
    """Ripple sees every bin above DC; the band comes from its output.

    Several models read the grid's spacing and top, so slicing the input would
    change in-band values. The DC bin, which Ripple evaluates to NaN, is dropped
    from the input; that leaves the spacing intact.
    """
    n_samples = ripple_generator.n_samples
    assert n_samples is not None
    delta_f = 256.0 / n_samples
    n_grid = n_samples // 2
    kernel = Mock(return_value=jnp.ones((n_grid, 2), dtype=jnp.float64))
    object.__setattr__(ripple_generator, "_kernel", kernel)

    intensity = ripple_generator.generate_batch(_ripple_sources())

    kernel.assert_called_once()
    frequencies, events = kernel.call_args.args
    np.testing.assert_array_equal(
        np.asarray(frequencies), np.arange(1, n_grid + 1) * delta_f
    )
    assert set(events) >= {"M_c", "eta", "d_L", "iota"}
    np.testing.assert_array_equal(
        intensity, np.ones((ripple_generator.frequencies.size, 2))
    )


def test_ripple_generator_evaluates_a_non_uniform_grid_directly() -> None:
    """The log spacings hand Ripple the band's own grid and slice nothing.

    They are not tied to an FFT segment: no Nyquist grid is built, so the
    sampling frequency and the segment length play no part.
    """
    generator = RippleGenerator(
        WaveformMetadata.model_validate(
            _grid_settings(
                minimum_frequency=20.0,
                frequency_resolution=2.0,
                frequency_spacing="log",
            )
        )
    )
    frequencies = np.asarray(generator.frequencies)
    kernel = Mock(return_value=jnp.ones((frequencies.size, 2), dtype=jnp.float64))
    object.__setattr__(generator, "_kernel", kernel)

    intensity = generator.generate_batch(_ripple_sources())

    kernel.assert_called_once()
    np.testing.assert_array_equal(np.asarray(kernel.call_args.args[0]), frequencies)
    assert frequencies[0] == 20.0
    assert frequencies[-1] == 100.0
    assert np.ptp(np.diff(frequencies)) > 0.0
    assert intensity.shape == (frequencies.size, 2)
    assert generator.n_samples is None
    assert generator.segment_duration is None


def test_ripple_generate_batch_has_no_nan_gradient(
    ripple_generator: RippleGenerator,
) -> None:
    """No NaN reaches reverse mode, which masking the output could not fix.

    A NaN DC bin would leak into the VJP however the output is masked
    (``0 * NaN``); excluding it from the input keeps gradients finite.
    """
    sources = _ripple_sources()

    def total_intensity(distance: jax.Array) -> jax.Array:
        return ripple_generator.generate_batch(
            {**sources, "luminosity_distance": distance}
        ).sum()

    gradient = jax.grad(total_intensity)(
        jnp.asarray(sources["luminosity_distance"], dtype=jnp.float64)
    )

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.asarray(gradient) < 0.0)


@pytest.mark.integration
def test_ripple_generator_infers_df_from_its_own_5_smooth_grid() -> None:
    """A minimum frequency off Ripple's 5-smooth grid is rejected.

    At ``sampling_frequency=1234.0, frequency_resolution=1.0`` Ripple's
    ``_next_smooth_even`` gives ``n=1250`` and ``delta_f=0.9872``, so
    ``minimum_frequency=2.0`` is off-grid; only an on-grid frequency is accepted.
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
    assert np.diff(np.asarray(frequencies)) == pytest.approx(0.9872)


# --------------------------------------------------------------------- #
# The grid rule
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


@pytest.mark.parametrize("minimum", [1, 7, 1234, 8193])
def test_next_smooth_even_is_the_smallest_admissible_length(minimum: int) -> None:
    """Ripple sizes its segment this way, so astrogwb has to agree bin for bin."""
    result = next_smooth_even(minimum)

    assert _is_smooth_even(result)
    assert result >= max(minimum, 2)
    assert not any(
        _is_smooth_even(candidate) for candidate in range(max(minimum, 2), result)
    )


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


def test_generate_batch_does_not_validate_values(
    ripple_generator: RippleGenerator,
) -> None:
    """``generate_batch`` lets through what ``check_sources`` rejects.

    Value checks would need a host sync on every chunk, breaking traced use.
    """
    sources = _ripple_sources() | {"lambda_1": np.array([-1.0, 0.0])}

    with pytest.raises(ValueError, match="non-negative"):
        ripple_generator.check_sources(sources)

    intensity = ripple_generator.generate_batch(sources)
    assert intensity.shape == (ripple_generator.frequencies.size, 2)


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ({"detector_frame_mass_1": np.array([1.4])}, "missing required"),
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


def _source_frame_sources(redshift: np.ndarray) -> dict[str, np.ndarray]:
    """``_ripple_sources`` with its masses restated in the source frame."""
    sources = _ripple_sources()
    scale = 1.0 + redshift
    return {
        "source_frame_mass_1": sources.pop("detector_frame_mass_1") / scale,
        "source_frame_mass_2": sources.pop("detector_frame_mass_2") / scale,
        "redshift": redshift,
        **sources,
    }


def test_source_frame_masses_and_redshift_give_the_detector_frame_intensity(
    ripple_generator: RippleGenerator,
) -> None:
    """Either form of the masses reaches the same waveform."""
    derived = _source_frame_sources(np.array([0.1, 0.5]))

    np.testing.assert_allclose(
        ripple_generator.generate_batch(derived),
        ripple_generator.generate_batch(_ripple_sources()),
    )


def test_explicit_detector_frame_masses_take_precedence(
    ripple_generator: RippleGenerator,
) -> None:
    """A stray source-frame column does not override a given detector-frame one."""
    sources = _ripple_sources() | {
        "source_frame_mass_1": np.array([9.0, 9.0]),
        "source_frame_mass_2": np.array([9.0, 9.0]),
        "redshift": np.array([0.3, 0.3]),
    }

    np.testing.assert_allclose(
        ripple_generator.generate_batch(sources),
        ripple_generator.generate_batch(_ripple_sources()),
    )


# --------------------------------------------------------------------- #
# Parity against gwmock-signal's GPL Ripple backend
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
    """Frequency axis and in-band intensity from ``gwmock_signal.RippleBackend``."""
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
    intensity = (
        jnp.abs(polarizations.plus[:, mask]) ** 2
        + jnp.abs(polarizations.cross[:, mask]) ** 2
    ).T
    return np.asarray(polarizations.frequencies[mask]), np.asarray(intensity)


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
def test_intensity_matches_the_gwmock_backend(approximant: str) -> None:
    """Families supported by both adapters agree with gwmock-signal's backend.

    Exact on the frequency axis. On intensity, ``rtol=1e-11`` with ``atol=0``: the
    residual is XLA fusion, at most ~2e-13 (``IMRPhenomHM``).
    """
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
    frequencies, intensity = _gwmock_reference(
        approximant, sources, segment_duration=1.0, **settings
    )

    # Jitted, like every real caller: the generator carries no jit of its own,
    # and the precessing models are far too slow to evaluate op by op.
    ours = jax.jit(generator.generate_batch)(sources)

    np.testing.assert_array_equal(np.asarray(generator.frequencies), frequencies)
    np.testing.assert_allclose(np.asarray(ours), intensity, rtol=1e-11, atol=0.0)


@pytest.mark.integration
@pytest.mark.parametrize("approximant", SUPPORTED_APPROXIMANTS)
def test_log_intensity_agrees_with_a_dense_uniform_grid(approximant: str) -> None:
    """Handing Ripple a log grid instead of the FFT grid changes no intensity.

    The reference is the same kernel on the union of a dense 0.25 Hz lattice and
    the log points: no interpolation, a different ``f[1] - f[0]`` and the
    same grid top, so any dependence on the grid shape would show.
    """
    band = {
        "approximant": approximant,
        "sampling_frequency": 2048.0,
        "minimum_frequency": 16.0,
        "maximum_frequency": 1024.0,
        "reference_frequency": 16.0,
    }
    sources = _parity_sources(approximant)
    dense = RippleGenerator(
        WaveformMetadata.model_validate({**band, "frequency_resolution": 0.25})
    )
    log = RippleGenerator(
        WaveformMetadata.model_validate(
            {**band, "frequency_spacing": "log", "frequency_resolution": 1.0}
        )
    )
    frequencies = np.asarray(log.frequencies)
    union = np.union1d(np.asarray(dense.frequencies), frequencies)
    assert union[1] - union[0] == 0.25
    assert union[-1] == frequencies[-1]

    intensity = np.asarray(jax.jit(log.generate_batch)(sources))
    union_intensity = np.asarray(
        jax.jit(dense._kernel)(
            jnp.asarray(union), ripple_parameters(approximant, sources)
        )
    )

    assert np.all(np.isfinite(intensity))
    assert np.all(intensity >= 0.0)
    np.testing.assert_allclose(
        intensity,
        union_intensity[np.searchsorted(union, frequencies)],
        rtol=1e-11,
        atol=0.0,
    )


@pytest.mark.integration
def test_untapered_nrtidal_intensity_survives_the_normal_merger_cutoff() -> None:
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

    frequencies, tapered_intensity = tapered(sources)
    _, untapered_intensity = untapered(sources)
    high_frequency = np.asarray(frequencies) > 3000.0

    assert np.all(np.asarray(tapered_intensity)[high_frequency] == 0.0)
    assert np.all(np.asarray(untapered_intensity)[high_frequency] > 0.0)


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
