"""Tests for asgwb.waveform module."""

from __future__ import annotations

import numpy as np
import pytest

from asgwb.waveform import (
    FrequencyGrid,
    WaveformBackend,
    WaveformGenerator,
    WaveformPolarizations,
    resolve_frequency_bounds,
)
from asgwb.waveform._bilby import BilbyWaveformBackend


@pytest.fixture
def grid() -> FrequencyGrid:
    return FrequencyGrid(
        duration=4.0,
        sampling_frequency=2048.0,
        minimum_frequency=20.0,
        maximum_frequency=1024.0,
        reference_frequency=50.0,
    )


@pytest.fixture
def bns_params() -> dict[str, float]:
    return {
        "mass_1": 1.4,
        "mass_2": 1.4,
        "luminosity_distance": 100.0,
        "theta_jn": 0.0,
        "psi": 0.0,
        "phase": 0.0,
        "geocent_time": 0.0,
        "ra": 0.0,
        "dec": 0.0,
        "lambda_1": 400.0,
        "lambda_2": 400.0,
        "chi_1": 0.0,
        "chi_2": 0.0,
    }


@pytest.fixture
def bbh_params() -> dict[str, float]:
    return {
        "mass_1": 30.0,
        "mass_2": 30.0,
        "luminosity_distance": 500.0,
        "theta_jn": 0.0,
        "psi": 0.0,
        "phase": 0.0,
        "geocent_time": 0.0,
        "ra": 0.0,
        "dec": 0.0,
        "a_1": 0.0,
        "a_2": 0.0,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
    }


class TestFrequencyGrid:
    def test_resolve_frequency_bounds_defaults_to_nyquist(self):
        min_frequency, max_frequency = resolve_frequency_bounds(2048.0)
        assert min_frequency == pytest.approx(10.0)
        assert max_frequency == pytest.approx(1024.0)

    def test_resolve_frequency_bounds_explicit_maximum(self):
        min_frequency, max_frequency = resolve_frequency_bounds(
            sampling_frequency=2048.0,
            minimum_frequency=10.0,
            maximum_frequency=800.0,
        )
        assert min_frequency == pytest.approx(10.0)
        assert max_frequency == pytest.approx(800.0)

    def test_resolve_frequency_bounds_invalid_maximum(self):
        with pytest.raises(ValueError, match="Nyquist"):
            resolve_frequency_bounds(
                sampling_frequency=2048.0,
                minimum_frequency=10.0,
                maximum_frequency=1200.0,
            )

    def test_construction(self):
        grid = FrequencyGrid(
            duration=8.0,
            sampling_frequency=2048.0,
            minimum_frequency=20.0,
            maximum_frequency=1024.0,
            reference_frequency=50.0,
        )
        assert grid.duration == 8.0
        assert grid.sampling_frequency == 2048.0
        assert grid.minimum_frequency == 20.0
        assert grid.maximum_frequency == 1024.0
        assert grid.reference_frequency == 50.0

    def test_construction_defaults_frequency_bounds(self):
        grid = FrequencyGrid(
            duration=8.0,
            sampling_frequency=2048.0,
            reference_frequency=50.0,
        )
        assert grid.minimum_frequency == 10.0
        assert grid.maximum_frequency == pytest.approx(1024.0)

    def test_frequencies_array(self, grid: FrequencyGrid):
        nyquist_frequency = grid.sampling_frequency / 2.0
        assert grid.frequencies[0] == pytest.approx(0.0)
        assert grid.frequencies[-1] == pytest.approx(nyquist_frequency)
        assert grid.in_band_frequencies[0] == pytest.approx(grid.minimum_frequency)
        assert grid.in_band_frequencies[-1] == pytest.approx(grid.maximum_frequency)

    def test_frequencies_array_matches_bilby(self, grid: FrequencyGrid):
        generator = WaveformGenerator(
            approximant="IMRPhenomPV2_NRTidalv2", grid=grid, source_type="BNS"
        )
        ours = generator.grid.frequencies
        theirs = generator.as_bilby_waveform_generator().frequency_array
        np.testing.assert_allclose(ours, theirs)

    def test_frozen(self):
        grid = FrequencyGrid(
            duration=8.0,
            sampling_frequency=2048.0,
            minimum_frequency=20.0,
            maximum_frequency=1024.0,
            reference_frequency=50.0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            grid.duration = 4.0  # type: ignore[misc]

    def test_min_ge_max_raises(self):
        with pytest.raises(ValueError, match="minimum_frequency"):
            FrequencyGrid(
                duration=8.0,
                sampling_frequency=2048.0,
                minimum_frequency=1024.0,
                maximum_frequency=20.0,
                reference_frequency=50.0,
            )

    def test_equal_min_max_raises(self):
        with pytest.raises(ValueError, match="minimum_frequency"):
            FrequencyGrid(
                duration=8.0,
                sampling_frequency=2048.0,
                minimum_frequency=100.0,
                maximum_frequency=100.0,
                reference_frequency=50.0,
            )

    def test_negative_duration_raises(self):
        with pytest.raises(ValueError, match="duration"):
            FrequencyGrid(
                duration=-1.0,
                sampling_frequency=2048.0,
                minimum_frequency=20.0,
                maximum_frequency=1024.0,
                reference_frequency=50.0,
            )

    def test_non_positive_sampling_frequency_raises(self):
        with pytest.raises(ValueError, match="sampling_frequency"):
            FrequencyGrid(
                duration=8.0,
                sampling_frequency=-2048.0,
                minimum_frequency=20.0,
                maximum_frequency=1024.0,
                reference_frequency=50.0,
            )

    def test_negative_min_frequency_raises(self):
        with pytest.raises(ValueError, match="minimum_frequency"):
            FrequencyGrid(
                duration=8.0,
                sampling_frequency=2048.0,
                minimum_frequency=-10.0,
                maximum_frequency=1024.0,
                reference_frequency=50.0,
            )

    def test_resample_same_grid_returns_values(self):
        source = FrequencyGrid(
            duration=4.0,
            sampling_frequency=8.0,
            minimum_frequency=0.0,
            maximum_frequency=4.0,
            reference_frequency=1.0,
        )
        x = np.arange(source.frequencies.size, dtype=np.float64)

        y = source.resample(x, source)

        np.testing.assert_allclose(y, x)

    def test_resample_to_grid_with_twice_frequency_spacing(self):
        source = FrequencyGrid(
            duration=4.0,
            sampling_frequency=8.0,
            minimum_frequency=0.0,
            maximum_frequency=4.0,
            reference_frequency=1.0,
        )
        target = FrequencyGrid(
            duration=2.0,
            sampling_frequency=8.0,
            minimum_frequency=0.0,
            maximum_frequency=4.0,
            reference_frequency=1.0,
        )
        x = np.arange(source.frequencies.size, dtype=np.float64)

        y = target.resample(x, source)

        np.testing.assert_allclose(y, x[::2])

    def test_resample_to_grid_with_half_frequency_spacing(self):
        source = FrequencyGrid(
            duration=2.0,
            sampling_frequency=8.0,
            minimum_frequency=0.0,
            maximum_frequency=4.0,
            reference_frequency=1.0,
        )
        target = FrequencyGrid(
            duration=4.0,
            sampling_frequency=8.0,
            minimum_frequency=0.0,
            maximum_frequency=4.0,
            reference_frequency=1.0,
        )
        x = np.arange(source.frequencies.size, dtype=np.float64)

        y = target.resample(x, source)

        np.testing.assert_allclose(y[::2], x)

    def test_resample_masks_values_outside_source_band_with_nan(self):
        source = FrequencyGrid(
            duration=4.0,
            sampling_frequency=8.0,
            minimum_frequency=1.0,
            maximum_frequency=3.0,
            reference_frequency=1.0,
        )
        target = FrequencyGrid(
            duration=4.0,
            sampling_frequency=8.0,
            minimum_frequency=0.5,
            maximum_frequency=3.5,
            reference_frequency=1.0,
        )
        x = source.frequencies.astype(np.float64, copy=True)

        y = target.resample(x, source)
        overlap_mask = (target.frequencies >= 1.0) & (target.frequencies <= 3.0)
        left_mask = target.in_band_mask & (target.frequencies < 1.0)
        right_mask = target.in_band_mask & (target.frequencies > 3.0)

        np.testing.assert_allclose(y[overlap_mask], target.frequencies[overlap_mask])
        assert np.isnan(y[left_mask]).all()
        assert np.isnan(y[right_mask]).all()

    def test_resample_complex_values_preserves_complex_dtype(self):
        source = FrequencyGrid(
            duration=4.0,
            sampling_frequency=8.0,
            minimum_frequency=1.0,
            maximum_frequency=3.0,
            reference_frequency=1.0,
        )
        target = FrequencyGrid(
            duration=4.0,
            sampling_frequency=8.0,
            minimum_frequency=0.5,
            maximum_frequency=3.5,
            reference_frequency=1.0,
        )
        x = source.frequencies.astype(np.complex128) + 1j * np.arange(
            source.frequencies.size, dtype=np.float64
        )

        y = target.resample(x, source)
        overlap_mask = (target.frequencies >= 1.0) & (target.frequencies <= 3.0)
        outside_mask = target.in_band_mask & ~overlap_mask

        assert y.dtype == np.complex128
        np.testing.assert_allclose(
            y[overlap_mask],
            x[overlap_mask],
        )
        assert np.isnan(y.real[outside_mask]).all()
        assert np.isnan(y.imag[outside_mask]).all()

    def test_resample_rejects_non_1d_input(self, grid: FrequencyGrid):
        with pytest.raises(ValueError, match="1D"):
            grid.resample(np.zeros((2, 2), dtype=np.float64), grid)

    def test_resample_rejects_shape_mismatch(self, grid: FrequencyGrid):
        with pytest.raises(ValueError, match="shape"):
            grid.resample(np.zeros(grid.frequencies.size - 1, dtype=np.float64), grid)


class TestWaveformPolarizations:
    def test_construction(self):
        grid = FrequencyGrid(
            duration=8.0,
            sampling_frequency=2048.0,
            minimum_frequency=20.0,
            maximum_frequency=1024.0,
            reference_frequency=50.0,
        )
        n = len(grid.frequencies)
        plus = np.zeros(n, dtype=np.complex128)
        cross = np.zeros(n, dtype=np.complex128)
        pol = WaveformPolarizations(grid=grid, plus=plus, cross=cross)
        assert pol.grid is grid
        assert pol.plus is plus
        assert pol.cross is cross


class TestBilbyWaveformBackend:
    def test_satisfies_protocol(self):
        grid = FrequencyGrid(
            duration=8.0,
            sampling_frequency=2048.0,
            minimum_frequency=20.0,
            maximum_frequency=1024.0,
            reference_frequency=50.0,
        )
        backend = BilbyWaveformBackend("IMRPhenomPV2_NRTidalv2", grid, "BNS")
        assert isinstance(backend, WaveformBackend)


class TestWaveformGenerator:
    def test_from_sampling_builds_grid(self):
        gen = WaveformGenerator.from_sampling(
            approximant="IMRPhenomPV2_NRTidalv2",
            duration=8.0,
            sampling_frequency=2048.0,
            reference_frequency=50.0,
            source_type="BNS",
        )
        assert gen.grid.minimum_frequency == pytest.approx(10.0)
        assert gen.grid.maximum_frequency == pytest.approx(1024.0)


@pytest.mark.integration
class TestBilbyWaveformBackendIntegration:
    def test_bns(self, grid: FrequencyGrid, bns_params: dict[str, float]):
        backend = BilbyWaveformBackend("IMRPhenomPV2_NRTidalv2", grid, "BNS")
        pol = backend.frequency_domain_polarizations(bns_params)
        assert isinstance(pol, WaveformPolarizations)
        assert pol.plus.shape == pol.cross.shape
        assert pol.plus.dtype == np.complex128

    def test_bbh(self, grid: FrequencyGrid, bbh_params: dict[str, float]):
        backend = BilbyWaveformBackend("IMRPhenomXP", grid, "BBH")
        pol = backend.frequency_domain_polarizations(bbh_params)
        assert isinstance(pol, WaveformPolarizations)
        assert pol.plus.shape == pol.cross.shape

    def test_caches_generator(self, grid: FrequencyGrid):
        """The bilby WaveformGenerator is constructed only once."""
        backend = BilbyWaveformBackend("IMRPhenomPV2_NRTidalv2", grid, "BNS")
        gen1 = backend.waveform_generator
        gen2 = backend.waveform_generator
        assert gen1 is gen2


@pytest.mark.integration
class TestWaveformGeneratorIntegration:
    def test_end_to_end(self, grid: FrequencyGrid, bns_params: dict[str, float]):
        gen = WaveformGenerator(
            approximant="IMRPhenomPV2_NRTidalv2",
            grid=grid,
            source_type="BNS",
        )
        pol = gen.frequency_domain_polarizations(bns_params)
        assert isinstance(pol, WaveformPolarizations)
        assert pol.grid is grid
        assert pol.plus.shape == pol.cross.shape
