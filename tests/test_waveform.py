"""Tests for asgwb.waveform module."""

from __future__ import annotations

import numpy as np
import pytest

from asgwb.waveform import (
    FrequencyGrid,
    WaveformBackend,
    WaveformGenerator,
    WaveformPolarizations,
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

    def test_frequencies_array(self):
        grid = FrequencyGrid(
            duration=4.0,
            sampling_frequency=2048.0,
            minimum_frequency=10.0,
            maximum_frequency=20.0,
            reference_frequency=50.0,
        )
        freqs = grid.frequencies
        delta_f = 1.0 / 4.0  # = 0.25
        expected = np.arange(10.0, 20.0 + delta_f, delta_f, dtype=np.float64)
        np.testing.assert_array_almost_equal(freqs, expected)
        assert freqs[0] == pytest.approx(10.0)

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


requires_bilby = pytest.mark.skipif(
    pytest.importorskip("bilby", reason="bilby not installed") is None,
    reason="bilby not installed",
)


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
