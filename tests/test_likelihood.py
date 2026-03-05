from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import asgwb.likelihood.likelihood as likelihood_module
from asgwb.likelihood.likelihood import SGWBGaussianLikelihood


class FakePrior:
    def __init__(self, samples: dict[str, np.ndarray]):
        self._samples = samples

    def keys(self):
        return self._samples.keys()

    def sample(self, n: int) -> dict[str, np.ndarray]:
        expected_n = len(next(iter(self._samples.values())))
        assert n == expected_n
        return self._samples


class DummyWaveformGenerator:
    def __init__(self, frequencies: np.ndarray):
        self.grid = SimpleNamespace(frequencies=frequencies)

    def frequency_domain_polarizations(self, parameters: dict[str, float]):
        raise AssertionError("frequency_domain_polarizations should not be called")


class DummyDetector:
    def __init__(self, psd_values: np.ndarray):
        self._psd_values = psd_values

    def psd(self, frequencies: np.ndarray) -> np.ndarray:
        assert len(frequencies) == len(self._psd_values)
        return self._psd_values


def _make_likelihood(
    *,
    frequencies: np.ndarray,
    prior_samples: dict[str, np.ndarray],
    mc_integral_npoints: int,
    detectors: list[DummyDetector] | None = None,
) -> SGWBGaussianLikelihood:
    return SGWBGaussianLikelihood(
        waveform_generator=DummyWaveformGenerator(frequencies),
        detectors=detectors or [],
        fiducial_spectral_density_generator=lambda f: np.zeros_like(f),
        intrinsic_prior_dict_generator=lambda _parameters: FakePrior(prior_samples),
        mc_integral_npoints=mc_integral_npoints,
    )


def test_spectral_density_iterates_over_samples(monkeypatch: pytest.MonkeyPatch):
    frequencies = np.array([10.0, 20.0, 30.0])
    samples = {
        "mass_1": np.array([1.0, 2.0, 3.0]),
        "chi_1": np.array([10.0, 20.0, 30.0]),
        "redshift": np.array([0.1, 0.2, 0.3]),
    }
    likelihood = _make_likelihood(
        frequencies=frequencies,
        prior_samples=samples,
        mc_integral_npoints=3,
    )

    monkeypatch.setattr(likelihood, "cosmology", lambda _parameters: object())
    monkeypatch.setattr(
        likelihood,
        "gravitational_wave_distance",
        lambda z, _parameters, _cosmology: np.ones_like(z),
    )
    monkeypatch.setattr(
        likelihood,
        "energy_flux",
        lambda injection: np.full_like(
            frequencies, injection["mass_1"] + 0.01 * injection["chi_1"]
        ),
    )

    result = likelihood.spectral_density(parameters={})
    expected = np.full_like(frequencies, 0.4 * np.mean([1.1, 2.2, 3.3]))
    np.testing.assert_allclose(result, expected)


def test_spectral_density_is_mc_normalized(monkeypatch: pytest.MonkeyPatch):
    frequencies = np.array([10.0, 20.0, 30.0])
    likelihood_small = _make_likelihood(
        frequencies=frequencies,
        prior_samples={
            "mass_1": np.array([1.0, 3.0]),
            "redshift": np.array([0.1, 0.2]),
        },
        mc_integral_npoints=2,
    )
    likelihood_large = _make_likelihood(
        frequencies=frequencies,
        prior_samples={
            "mass_1": np.array([1.0, 3.0, 1.0, 3.0]),
            "redshift": np.array([0.1, 0.2, 0.1, 0.2]),
        },
        mc_integral_npoints=4,
    )

    for likelihood in (likelihood_small, likelihood_large):
        monkeypatch.setattr(likelihood, "cosmology", lambda _parameters: object())
        monkeypatch.setattr(
            likelihood,
            "gravitational_wave_distance",
            lambda z, _parameters, _cosmology: np.ones_like(z),
        )
        monkeypatch.setattr(
            likelihood,
            "energy_flux",
            lambda injection: np.full_like(frequencies, injection["mass_1"]),
        )

    result_small = likelihood_small.spectral_density(parameters={})
    result_large = likelihood_large.spectral_density(parameters={})
    np.testing.assert_allclose(result_small, result_large)


def test_inverse_covariance_uses_upper_triangle_detector_axes(
    monkeypatch: pytest.MonkeyPatch,
):
    frequencies = np.array([10.0, 20.0, 30.0])
    detectors = [
        DummyDetector(np.array([1.0, 1.0, 1.0])),
        DummyDetector(np.array([2.0, 2.0, 2.0])),
        DummyDetector(np.array([4.0, 4.0, 4.0])),
    ]
    likelihood = _make_likelihood(
        frequencies=frequencies,
        prior_samples={"redshift": np.array([0.1])},
        mc_integral_npoints=1,
        detectors=detectors,
    )

    orf_pair = np.zeros((3, 3, 3))
    orf_pair[0, 1] = np.array([10.0, 20.0, 30.0])
    orf_pair[1, 0] = orf_pair[0, 1]
    orf_pair[0, 2] = np.array([1.0, 2.0, 3.0])
    orf_pair[2, 0] = orf_pair[0, 2]
    orf_pair[1, 2] = np.array([4.0, 5.0, 6.0])
    orf_pair[2, 1] = orf_pair[1, 2]

    monkeypatch.setattr(
        likelihood_module,
        "pairwise_overlap_reduction_function",
        lambda _frequencies, _detectors: orf_pair,
    )

    result = likelihood.inverse_covariance
    expected = np.array([5.75, 11.125, 16.5])
    assert result.shape == (3,)
    np.testing.assert_allclose(result, expected)


def test_log_likelihood_requires_parameters():
    frequencies = np.array([10.0, 20.0, 30.0])
    likelihood = _make_likelihood(
        frequencies=frequencies,
        prior_samples={"redshift": np.array([0.1])},
        mc_integral_npoints=1,
    )

    with pytest.raises(ValueError, match="parameters must be provided"):
        likelihood.log_likelihood(parameters=None)


def test_init_rejects_non_positive_mc_integral_npoints():
    frequencies = np.array([10.0, 20.0, 30.0])
    with pytest.raises(ValueError, match="mc_integral_npoints must be positive"):
        _make_likelihood(
            frequencies=frequencies,
            prior_samples={"redshift": np.array([0.1])},
            mc_integral_npoints=0,
        )

    with pytest.raises(ValueError, match="mc_integral_npoints must be positive"):
        _make_likelihood(
            frequencies=frequencies,
            prior_samples={"redshift": np.array([0.1])},
            mc_integral_npoints=-5,
        )
