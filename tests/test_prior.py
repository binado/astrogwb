"""Tests for the asgwb.prior module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from asgwb.prior.redshift import (
    AVAILABLE_TIME_DELAY_MODELS,
    InverseTimeDelayPdf,
    madau_dickinson_source_frame_distribution,
    power_law_source_frame_distribution,
    redshift_pdf,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def z_array():
    return np.linspace(0.01, 3.0, 300)


@pytest.fixture
def flat_lcdm():
    """Standard flat ΛCDM cosmology from astropy."""
    from astropy.cosmology import FlatLambdaCDM

    return FlatLambdaCDM(H0=67.4, Om0=0.315)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestPowerLawSourceFrameDistribution:
    def test_normalization_at_zero(self, z_array):
        """Should equal 1.0 at z=0."""
        z = np.concatenate([[0.0], z_array])
        result = power_law_source_frame_distribution(z, lamb=2.9)
        assert result[0] == pytest.approx(1.0)

    def test_monotone_increasing_for_positive_lamb(self, z_array):
        result = power_law_source_frame_distribution(z_array, lamb=2.9)
        assert np.all(np.diff(result) > 0)

    def test_lamb_zero_gives_ones(self, z_array):
        result = power_law_source_frame_distribution(z_array, lamb=0.0)
        np.testing.assert_allclose(result, np.ones_like(z_array))


class TestMadauDickinsonSourceFrameDistribution:
    def test_normalization_at_zero(self):
        z = np.array([0.0, 1.0, 2.0])
        result = madau_dickinson_source_frame_distribution(
            z, kappa=5.7, gamma=2.7, z_peak=2.0
        )
        assert result[0] == pytest.approx(1.0)

    def test_peaks_near_z_peak(self, z_array):
        z_peak = 2.0
        result = madau_dickinson_source_frame_distribution(
            z_array, kappa=5.7, gamma=2.7, z_peak=z_peak
        )
        peak_idx = np.argmax(result)
        assert abs(z_array[peak_idx] - z_peak) < 0.5


class TestInverseTimeDelayPdf:
    def test_zero_below_minimum(self):
        """Values at or below minimum_time_delay should be zero."""
        time_delay = np.array([[0.0, 0.01, 0.5], [0.0, 0.01, 0.5]])
        minimum_time_delay = 0.02
        result = InverseTimeDelayPdf(minimum_time_delay=minimum_time_delay).pdf(
            time_delay
        )
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0

    def test_non_negative(self):
        """Output should be non-negative everywhere."""
        # time_delay values spanning causal and non-causal regions
        time_delay = np.linspace(-1.0, 5.0, 100).reshape(10, 10)
        result = InverseTimeDelayPdf(minimum_time_delay=0.02).pdf(time_delay)
        assert np.all(result >= 0)

    def test_analytical_normalization(self):
        """A single-row input integrates to 1: ∫_{t_min}^{t_max} pdf dt ≈ 1."""
        t_min, t_max = 0.05, 10.0
        # Use a fine log-spaced grid to avoid discretization error near t_min.
        # Shape (1, N): one row whose max_time_delay is t_max.
        t = np.logspace(np.log10(t_min), np.log10(t_max), 2000)
        time_delay = t[np.newaxis, :]  # shape (1, 2000)
        pdf = InverseTimeDelayPdf(minimum_time_delay=t_min).pdf(time_delay)
        integral = trapezoid(pdf[0], x=t)
        assert integral == pytest.approx(1.0, rel=1e-3)

    def test_available_models_registry(self):
        assert "inverse_time_delay" in AVAILABLE_TIME_DELAY_MODELS
        model = AVAILABLE_TIME_DELAY_MODELS["inverse_time_delay"]
        assert hasattr(model, "pdf")
        assert callable(model.pdf)


class TestRedshiftPdf:
    def test_normalized_output(self, z_array, flat_lcdm):
        source_dist = power_law_source_frame_distribution(z_array, lamb=2.9)
        pdf = redshift_pdf(z_array, flat_lcdm, source_dist, time_delay_fn=None)
        integral = trapezoid(pdf, x=z_array)
        assert integral == pytest.approx(1.0, rel=1e-3)

    def test_with_time_delay(self, z_array, flat_lcdm):
        source_dist = power_law_source_frame_distribution(z_array, lamb=2.9)
        pdf = redshift_pdf(
            z_array,
            flat_lcdm,
            source_dist,
            time_delay_fn=InverseTimeDelayPdf(),
        )
        integral = trapezoid(pdf, x=z_array)
        assert integral == pytest.approx(1.0, rel=1e-3)

    def test_no_normalization(self, z_array, flat_lcdm):
        source_dist = power_law_source_frame_distribution(z_array, lamb=2.9)
        pdf = redshift_pdf(
            z_array, flat_lcdm, source_dist, time_delay_fn=None, normalize=False
        )
        # Unnormalized: integral need not equal 1, but pdf should be positive
        assert np.all(pdf >= 0)

    def test_z_min_z_max_masking(self, flat_lcdm):
        z = np.linspace(0.01, 4.0, 400)
        source_dist = power_law_source_frame_distribution(z, lamb=2.9)
        z_min, z_max = 0.5, 2.0
        pdf = redshift_pdf(
            z,
            flat_lcdm,
            source_dist,
            time_delay_fn=None,
            z_min=z_min,
            z_max=z_max,
        )
        assert np.all(pdf[z < z_min] == 0.0)
        assert np.all(pdf[z > z_max] == 0.0)


# ---------------------------------------------------------------------------
# Integration tests (require bilby + lal)
# ---------------------------------------------------------------------------


class TestPriorClasses:
    def test_uniform_source_frame_prior_sampling(self):
        from asgwb.prior import UniformSourceFramePrior

        prior = UniformSourceFramePrior(minimum=0.01, maximum=3.0, name="redshift")
        samples = prior.sample(100)
        assert len(samples) == 100
        assert np.all(samples >= 0.01)
        assert np.all(samples <= 3.0)

    def test_power_law_prior_sampling(self):
        from asgwb.prior import PowerLawRedshiftPrior

        prior = PowerLawRedshiftPrior(minimum=0.01, maximum=3.0, name="redshift")
        samples = prior.sample(100)
        assert len(samples) == 100

    def test_power_law_prior_custom_lamb(self):
        from asgwb.prior import PowerLawRedshiftPrior

        prior = PowerLawRedshiftPrior(
            minimum=0.01, maximum=3.0, name="redshift", lamb=1.5
        )
        assert prior.model_parameters["lamb"] == 1.5

    def test_madau_dickinson_prior_sampling(self):
        from asgwb.prior import MadauDickinsonRedshiftPrior

        prior = MadauDickinsonRedshiftPrior(minimum=0.01, maximum=3.0, name="redshift")
        samples = prior.sample(100)
        assert len(samples) == 100

    def test_madau_dickinson_prior_with_time_delay(self):
        from asgwb.prior import MadauDickinsonRedshiftPrior

        prior = MadauDickinsonRedshiftPrior(
            minimum=0.01,
            maximum=3.0,
            name="redshift",
            time_delay_fn="inverse_time_delay",
        )
        samples = prior.sample(50)
        assert len(samples) == 50

    def test_model_attributes(self):
        from asgwb.prior import MadauDickinsonRedshiftPrior, PowerLawRedshiftPrior

        assert PowerLawRedshiftPrior.model_type == "redshift"
        assert PowerLawRedshiftPrior.model_name == "power_law"
        assert MadauDickinsonRedshiftPrior.model_name == "madau_dickinson"

    def test_bilby_re_exports(self):
        from asgwb.prior import Gaussian, Uniform

        u = Uniform(minimum=0, maximum=1, name="test")
        g = Gaussian(mu=0, sigma=1, name="test")
        assert u.sample() is not None
        assert g.sample() is not None
