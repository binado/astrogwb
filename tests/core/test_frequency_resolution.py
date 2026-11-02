r"""Check catalog frequency grids and likelihood-ratio behavior under refinement.

Everything here rests on one property of the catalog grid, pinned by the first
test: it is ``f_min + df * arange(n)``, so ``frequencies[::k]`` of a fine
catalog is *exactly* the grid a ``k * df`` catalog would have been built on.
Coarsening by subsampling therefore holds the sources fixed and varies only
:math:`\Delta f`.

``FINE_DF`` is a negative power of two so that ``k * FINE_DF`` is exact in
binary and the two grids agree essentially to round-off.

Bin widths derive from the subsampled ``frequencies``; the band is a mask on
the full grid, never a slice.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    build_mock_catalog,
    build_reference_catalog,
    build_reference_spectrum,
)

from astrogwb.background import spectral_snr_squared
from astrogwb.background.catalog import (
    CatalogMetadata,
    PolarizationPowerData,
)
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import effective_psd, gaussian_bin_scale, load_detector
from astrogwb.frequency import frequency_mask

#: Reference resolution, and the band the refinement study runs over.
FINE_DF = 0.25
F_MIN = 2.0
F_MAX = 256.0

#: Where the reference catalog stops: above the detector-frame merger frequency
#: of a BNS at the lowest redshift, so no query of the rescaled spectrum is cut.
REFERENCE_F_MAX = 2048.0

#: Sources. The subject here is ``df``, not ``N``, so this is set by build cost.
NUM_SOURCES = 256

#: Grid-coarsening factors, giving df from 0.25 Hz up to 4 Hz.
SUBSAMPLE_FACTORS: tuple[int, ...] = (2, 4, 8, 16)

DETECTORS: tuple[str, ...] = ("E1", "E2", "E3")

OBSERVATION_TIME = 1.0

#: H0 the log-likelihood ratio is measured across. Far enough from the
#: fiducial that ``Delta log L`` is order 50 rather than order round-off.
OFFSET_H0 = 62.0

#: Relative log-likelihood-ratio residual tolerated at the first coarsening
#: step. This checks convergence of the ratio across frequency grids.
FIRST_STEP_TOLERANCE = 5e-3


@pytest.fixture(scope="module")
def fine_catalog(
    mock_population: dict[str, np.ndarray],
) -> tuple[PolarizationPowerData, CatalogMetadata]:
    """The reference catalog every coarser grid is subsampled from."""
    return build_mock_catalog(
        mock_population,
        num_sources=NUM_SOURCES,
        f_min=F_MIN,
        f_max=F_MAX,
        frequency_resolution=FINE_DF,
    )


def test_subsampling_a_fine_catalog_matches_a_coarse_one(
    mock_population: dict[str, np.ndarray],
    fine_catalog: tuple[PolarizationPowerData, CatalogMetadata],
) -> None:
    """``[::k]`` of a fine catalog *is* the catalog built at ``k * df``.

    The premise of every other test in this module.
    """
    fine_data, _ = fine_catalog
    fine_frequencies = np.asarray(fine_data["frequencies"])
    fine_power = np.asarray(fine_data["polarization_power"])

    for factor in SUBSAMPLE_FACTORS:
        coarse, _ = build_mock_catalog(
            mock_population,
            num_sources=NUM_SOURCES,
            f_min=F_MIN,
            f_max=F_MAX,
            frequency_resolution=factor * FINE_DF,
        )
        np.testing.assert_allclose(
            fine_frequencies[::factor], np.asarray(coarse["frequencies"])
        )
        np.testing.assert_allclose(
            fine_power[::factor], np.asarray(coarse["polarization_power"])
        )


def _analysis_at(frequencies: jax.Array) -> dict[str, Any]:
    """Derive the masked analysis inputs on one observed grid.

    The PSD and mask are rebuilt on each grid; only the sources are shared.
    ``noise_scale`` is one wherever the mask excludes a bin.
    """
    network_psd = jnp.asarray(
        effective_psd(
            np.asarray(frequencies), [load_detector(name) for name in DETECTORS]
        )
    )
    mask = frequency_mask(frequencies, fmin=F_MIN, fmax=F_MAX) & jnp.isfinite(
        network_psd
    )
    noise_scale = gaussian_bin_scale(network_psd, OBSERVATION_TIME, frequencies)
    return {
        "frequencies": frequencies,
        "mask": mask,
        "effective_psd": network_psd,
        "noise_scale": jnp.where(mask, noise_scale, 1.0),
        "num_bins": int(jnp.sum(mask)),
    }


@pytest.fixture(scope="module")
def reference() -> tuple[PolarizationPowerData, CatalogMetadata]:
    """One reference catalog every observed grid is predicted from.

    It reaches past the detector-frame merger frequency at the window's lower
    edge, which the rescaled spectrum's queries above the band need.
    """
    return build_reference_catalog(
        NUM_SOURCES,
        f_min=F_MIN,
        f_max=REFERENCE_F_MAX,
        frequency_resolution=FINE_DF,
    )


@pytest.fixture(scope="module")
def resolutions(
    reference: tuple[PolarizationPowerData, CatalogMetadata],
) -> dict[int, dict[str, Any]]:
    """Masked analysis inputs, mock observations, and SNR^2 at each resolution."""
    data, _ = reference
    fine_frequencies = jnp.asarray(data["frequencies"])
    fine_frequencies = fine_frequencies[fine_frequencies <= F_MAX]

    runs: dict[int, dict[str, Any]] = {}
    for factor in (1, *SUBSAMPLE_FACTORS):
        run = _analysis_at(fine_frequencies[::factor])
        # One estimator per observed grid, all predicting from the same
        # sources. The injection is its own fiducial prediction, so the
        # chi^2 term vanishes at the fiducials and every log-weight is zero.
        run["estimator"] = build_reference_spectrum(
            *reference, frequencies=run["frequencies"]
        )
        run["observed"] = run["estimator"].spectrum(FIDUCIALS)
        run["snr_squared"] = float(
            spectral_snr_squared(
                run["observed"],
                run["effective_psd"],
                OBSERVATION_TIME * SECONDS_PER_YEAR,
                run["frequencies"],
                frequency_mask=run["mask"],
            )
        )
        runs[factor] = run
    return runs


def _log_likelihood(run: dict[str, Any], hubble_constant: float) -> float:
    """Gaussian log-density of the injection under the H0-shifted template."""
    model = run["estimator"].spectrum({**FIDUCIALS, "H0": hubble_constant})
    log_prob = dist.Normal(model, run["noise_scale"]).log_prob(run["observed"])
    return float(jnp.sum(jnp.where(run["mask"], log_prob, 0.0)))


def test_the_absolute_log_likelihood_does_not_converge(
    resolutions: dict[int, dict[str, Any]],
) -> None:
    r"""Only differences at fixed resolution mean anything.

    The Gaussian bin scale is :math:`\sigma_i = S_{\mathrm{eff},i}/
    \sqrt{2T\Delta f}`, so coarsening by ``k`` grows every :math:`\sigma_i` by
    :math:`\sqrt{k}` while dropping the bin count by ``k``. The normalization
    :math:`-\tfrac{1}{2}\sum_i\log(2\pi\sigma_i^2)` therefore scales with the
    number of bins and converges to nothing.
    """
    at_fiducial = {
        factor: _log_likelihood(run, FIDUCIALS["H0"])
        for factor, run in resolutions.items()
    }
    # It tracks the bin count, so the coarsest grid is off by an order of
    # magnitude while the log-likelihood ratio is off by 11%.
    bin_ratio = (
        resolutions[1]["num_bins"] / resolutions[SUBSAMPLE_FACTORS[-1]]["num_bins"]
    )
    np.testing.assert_allclose(
        at_fiducial[1] / at_fiducial[SUBSAMPLE_FACTORS[-1]], bin_ratio, rtol=0.05
    )
    assert at_fiducial[1] / at_fiducial[SUBSAMPLE_FACTORS[-1]] > 10.0


def test_the_log_likelihood_ratio_residual_is_the_snr_squared_residual(
    resolutions: dict[int, dict[str, Any]],
) -> None:
    r"""The two convergence curves are the same curve, exactly.

    The catalog is its own importance proposal, so at the fiducials the
    injection *is* the template and the :math:`\chi^2` term is identically
    zero. With only :math:`H_0` varied the predicted spectrum is
    :math:`A\,S_h(H_0^{\rm fid})` with :math:`A = H_0^{\rm fid}/H_0`, hence

    .. math::

        \Delta\log\mathcal{L}(H_0) = -\tfrac{1}{2}(1 - A)^2\rho^2 ,

    and the relative residual of :math:`\Delta\log\mathcal{L}` under
    coarsening must equal that of :math:`\rho^2`, with no dependence on
    :math:`H_0` at all. This ties ``spectral_density``,
    ``spectral_snr_squared`` and ``gaussian_bin_scale`` to one another.
    """
    at_fiducial = {
        factor: _log_likelihood(run, FIDUCIALS["H0"])
        for factor, run in resolutions.items()
    }
    for factor in SUBSAMPLE_FACTORS:
        run = resolutions[factor]
        delta_residual = (_log_likelihood(run, OFFSET_H0) - at_fiducial[factor]) / (
            _log_likelihood(resolutions[1], OFFSET_H0) - at_fiducial[1]
        ) - 1.0
        snr_squared_residual = run["snr_squared"] / resolutions[1]["snr_squared"] - 1.0
        np.testing.assert_allclose(delta_residual, snr_squared_residual, rtol=1e-5)
    # The ratio converges: the first coarsening moves it very little.
    first = resolutions[SUBSAMPLE_FACTORS[0]]
    assert (
        abs(first["snr_squared"] / resolutions[1]["snr_squared"] - 1.0)
        < FIRST_STEP_TOLERANCE
    )
