r"""Check catalog frequency grids and likelihood-ratio behavior under refinement.

Everything here rests on one property of the catalog grid, pinned by the first
test: it is ``f_min + df * arange(n)``, so ``frequencies[::k]`` of a fine
catalog is *exactly* the grid a ``k * df`` catalog would have been built on.
Coarsening by subsampling therefore holds the sources fixed and varies only
:math:`\Delta f`.

``FINE_DF`` is a negative power of two so that ``k * FINE_DF`` is exact in
binary and the two grids agree essentially to round-off.

Nothing here tracks a bin width by hand: the SNR and the noise scale take the
subsampled ``frequencies`` and derive each bin's width from them, so a coarser
grid is coarser everywhere at once. The band mask is passed alongside the full
grid rather than used to slice it, because widths belong to the whole axis.

The band stops at 256 Hz to keep the grid manageable while retaining the
signal contribution used by ``notebooks/catalog_convergence.py``.

Fast by design -- no NUTS, so these are not marked ``integration``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import (
    FIDUCIALS,
    build_mock_catalog,
    catalog_samples,
    mock_merger_rate_fn,
    mock_target_model,
)

from astrogwb.catalog import PolarizationPowerCatalog
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import effective_psd, gaussian_bin_scale, load_sensitivity_map
from astrogwb.frequency import frequency_mask
from astrogwb.gwb import spectral_density, spectral_snr_squared
from astrogwb.importance.spectral import build_importance_spectrum
from astrogwb.populations import DEFAULT_DENSITY_SITES

#: Reference resolution, and the band the refinement study runs over.
FINE_DF = 0.25
F_MIN = 2.0
F_MAX = 256.0

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
def fine_catalog(mock_population: dict[str, np.ndarray]) -> PolarizationPowerCatalog:
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
    fine_catalog: PolarizationPowerCatalog,
) -> None:
    """``[::k]`` of a fine catalog *is* the catalog built at ``k * df``.

    The premise of every other test in this module, and of
    ``notebooks/catalog_convergence.py``. Both grids are formed from integer indices times a
    power-of-two ``df``, so any real disagreement means the grid construction
    changed.
    """
    fine_frequencies = np.asarray(fine_catalog.frequencies)
    fine_power = np.asarray(fine_catalog.polarization_power)

    for factor in SUBSAMPLE_FACTORS:
        coarse = build_mock_catalog(
            mock_population,
            num_sources=NUM_SOURCES,
            f_min=F_MIN,
            f_max=F_MAX,
            frequency_resolution=factor * FINE_DF,
        )
        np.testing.assert_allclose(
            fine_frequencies[::factor], np.asarray(coarse.frequencies)
        )
        np.testing.assert_allclose(
            fine_power[::factor], np.asarray(coarse.polarization_power)
        )


def _analysis_at(
    catalog: PolarizationPowerCatalog, factor: int, sensitivities: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-derive the masked analysis inputs on the grid coarsened by ``factor``.

    The PSD and the mask are rebuilt on the subsampled grid rather than
    subsampled themselves, so nothing about the coarse analysis is inherited
    from the fine one except the sources. Every array stays on the full
    subsampled grid and the band is a mask, so bin widths derive from the whole
    axis; ``noise_scale`` is a placeholder of one wherever the mask excludes a
    bin, which keeps an infinite PSD out of the log density.
    """
    frequencies = jnp.asarray(catalog.frequencies)[::factor]
    polarization_power = jnp.asarray(catalog.polarization_power)[::factor]

    network_psd = jnp.asarray(
        effective_psd(np.asarray(frequencies), DETECTORS, sensitivities)
    )
    mask = frequency_mask(frequencies, fmin=F_MIN, fmax=F_MAX) & jnp.isfinite(
        network_psd
    )
    noise_scale = gaussian_bin_scale(network_psd, OBSERVATION_TIME, frequencies)
    return {
        "frequencies": frequencies,
        "mask": mask,
        "polarization_power": polarization_power,
        "effective_psd": network_psd,
        "noise_scale": jnp.where(mask, noise_scale, 1.0),
        "num_bins": int(jnp.sum(mask)),
    }


@pytest.fixture(scope="module")
def resolutions(fine_catalog: PolarizationPowerCatalog) -> dict[int, dict[str, Any]]:
    """Masked analysis inputs, mock observations, and SNR^2 at each resolution."""
    samples = catalog_samples(fine_catalog)
    # The catalog is its own proposal: preparation caches the density and
    # reference distances the target re-forms at FIDUCIALS, which is what makes
    # every fiducial log-weight exactly zero.
    estimator, log_weights_fn = build_importance_spectrum(
        fine_catalog,
        source_model=mock_target_model(),
        merger_rate_fn=mock_merger_rate_fn(),
        density_sites=DEFAULT_DENSITY_SITES,
    )
    total_merger_rate = jnp.asarray(estimator(FIDUCIALS)[1]["total_merger_rate"])

    def weights_fn(params: dict[str, float]) -> tuple[jax.Array, jax.Array]:
        _, extras = estimator(params)
        return jnp.asarray(extras["total_merger_rate"]), log_weights_fn(params)

    # Loaded once: the noise curves are the same at every resolution, and
    # re-reading them per grid dominated the module's runtime.
    sensitivities = load_sensitivity_map(DETECTORS)

    runs: dict[int, dict[str, Any]] = {}
    for factor in (1, *SUBSAMPLE_FACTORS):
        run = _analysis_at(fine_catalog, factor, sensitivities)
        run["samples"] = samples
        run["weights_fn"] = weights_fn
        # The catalog is its own proposal, so the injection is the unweighted
        # contraction and every log-weight is exactly zero at the fiducials.
        run["observed"] = spectral_density(
            run["polarization_power"],
            jnp.ones(NUM_SOURCES),
            total_merger_rate,
            source_parameters=run["samples"],
        )
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
    total_merger_rate, log_weights = run["weights_fn"](
        {**FIDUCIALS, "H0": hubble_constant}
    )
    model = spectral_density(
        run["polarization_power"],
        jnp.exp(log_weights),
        total_merger_rate,
        source_parameters=run["samples"],
    )
    log_prob = dist.Normal(model, run["noise_scale"]).log_prob(run["observed"])
    return float(jnp.sum(jnp.where(run["mask"], log_prob, 0.0)))


def test_log_likelihood_ratio_converges_but_the_absolute_value_does_not(
    resolutions: dict[int, dict[str, Any]],
) -> None:
    r"""Only differences at fixed resolution mean anything.

    The Gaussian bin scale is :math:`\sigma_i = S_{\mathrm{eff},i}/
    \sqrt{2T\Delta f}`, so coarsening by ``k`` grows every :math:`\sigma_i` by
    :math:`\sqrt{k}` while dropping the bin count by ``k``. The normalization
    :math:`-\tfrac{1}{2}\sum_i\log(2\pi\sigma_i^2)` therefore scales with the
    number of bins and converges to nothing. This test is the executable form
    of that statement, so that the drift is never mistaken for a bug.
    """
    at_fiducial = {
        factor: _log_likelihood(run, FIDUCIALS["H0"])
        for factor, run in resolutions.items()
    }
    delta = {
        factor: _log_likelihood(run, OFFSET_H0) - at_fiducial[factor]
        for factor, run in resolutions.items()
    }

    # The ratio converges.
    residuals = {factor: delta[factor] / delta[1] - 1.0 for factor in SUBSAMPLE_FACTORS}
    magnitudes = [abs(residuals[factor]) for factor in SUBSAMPLE_FACTORS]
    assert magnitudes == sorted(magnitudes), residuals
    assert abs(residuals[SUBSAMPLE_FACTORS[0]]) < FIRST_STEP_TOLERANCE, residuals

    # The absolute value does not: it tracks the bin count, so the coarsest
    # grid is off by an order of magnitude while its ratio is off by 11%.
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
    :math:`H_0` at all. Asserting it ties ``spectral_density``,
    ``spectral_snr_squared``, and ``gaussian_bin_scale`` to one another: a
    factor lost in any one of them breaks the identity, whereas each function's
    own tests would still pass.
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


def test_every_log_weight_is_exactly_zero_at_the_fiducials(
    resolutions: dict[int, dict[str, Any]],
) -> None:
    """The premise the identity above rests on, stated on its own."""
    run = resolutions[1]
    _, log_weights = run["weights_fn"](dict(FIDUCIALS))
    assert isinstance(log_weights, jax.Array)
    np.testing.assert_array_equal(np.asarray(log_weights), 0.0)
