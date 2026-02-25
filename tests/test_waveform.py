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

from .conftest import requires_bilby


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
        assert min_frequency == pytest.approx(0.0)
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
        assert grid.minimum_frequency == 0.0
        assert grid.maximum_frequency == pytest.approx(1024.0)

    def test_frequencies_array(self, grid: FrequencyGrid):
        nyquist_frequency = grid.sampling_frequency / 2.0
        assert grid.frequencies[0] == pytest.approx(0.0)
        assert grid.frequencies[-1] == pytest.approx(nyquist_frequency)
        assert grid.in_band_frequencies[0] == pytest.approx(grid.minimum_frequency)
        assert grid.in_band_frequencies[-1] == pytest.approx(grid.maximum_frequency)

    @requires_bilby
    def test_frequencies_array_matches_bilby(self, grid: FrequencyGrid):
        from bilby.core.utils.series import create_frequency_series

        frequencies_bilby = create_frequency_series(
            grid.sampling_frequency, grid.duration
        )
        np.testing.assert_allclose(grid.frequencies, frequencies_bilby)

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
        hp = np.zeros(n, dtype=np.complex128)
        hc = np.zeros(n, dtype=np.complex128)
        pol = WaveformPolarizations(grid=grid, hp=hp, hc=hc)
        assert pol.grid is grid
        assert pol.hp is hp
        assert pol.hc is hc


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


@pytest.mark.integration
@requires_bilby
class TestBilbyWaveformBackendIntegration:
    def test_bns(self, grid: FrequencyGrid, bns_params: dict[str, float]):
        backend = BilbyWaveformBackend("IMRPhenomPV2_NRTidalv2", grid, "BNS")
        pol = backend.frequency_domain_polarizations(bns_params)
        assert isinstance(pol, WaveformPolarizations)
        assert pol.hp.shape == pol.hc.shape
        assert pol.hp.dtype == np.complex128

    def test_bbh(self, grid: FrequencyGrid, bbh_params: dict[str, float]):
        backend = BilbyWaveformBackend("IMRPhenomXP", grid, "BBH")
        pol = backend.frequency_domain_polarizations(bbh_params)
        assert isinstance(pol, WaveformPolarizations)
        assert pol.hp.shape == pol.hc.shape

    def test_caches_generator(self, grid: FrequencyGrid):
        """The bilby WaveformGenerator is constructed only once."""
        backend = BilbyWaveformBackend("IMRPhenomPV2_NRTidalv2", grid, "BNS")
        gen1 = backend._waveform_generator
        gen2 = backend._waveform_generator
        assert gen1 is gen2


@pytest.mark.integration
@requires_bilby
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
        assert pol.hp.shape == pol.hc.shape
