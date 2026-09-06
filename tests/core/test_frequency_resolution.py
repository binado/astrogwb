r"""How the answer depends on the catalog's frequency resolution.

Two independent discretizations meet in a catalog-based background, and only
one of them is about the number of sources. :func:`spectral_snr_squared` is a
Riemann sum

.. math::

    \mathrm{SNR}^2 = 2T\,\Delta f \sum_i S_{h,i}^2 / S_{\mathrm{eff},i}^2,

so refining :math:`\Delta f` is a convergence question in its own right, and
nothing else in the core suite asks it.

Everything here rests on one property of the catalog grid, pinned by the first
test: it is ``f_min + df * arange(n)``, so ``frequencies[::k]`` of a fine
catalog is *exactly* the grid a ``k * df`` catalog would have been built on.
Coarsening by subsampling therefore holds the sources fixed and varies only
:math:`\Delta f`.

``FINE_DF`` is a negative power of two so that ``k * FINE_DF`` is exact in
binary and the two grids agree bit for bit rather than to a tolerance.

The band stops at 256 Hz because that is where the signal is: with the ET
effective PSD the SNR integrand is a peak a few hertz wide near 7 Hz, and
99.9% of :math:`\rho^2` accumulates below 150 Hz. ``notebooks/catalog_convergence.py``
plots that integrand; it is the reason ``FINE_DF`` has to be this small to
serve as a converged reference at all.

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
    make_redshift_grid,
)

from astrogwb.catalog import Catalog
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.cosmology import log_gw_em_ratio
from astrogwb.detector import effective_psd, gaussian_bin_scale, load_sensitivity_map
from astrogwb.frequency import apply_frequency_mask, frequency_mask
from astrogwb.gwb import spectral_density, spectral_snr_squared
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    bns_population,
    compute_merger_rate_distance_and_logprob,
)
from astrogwb.importance.population import importance_log_weights

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

#: Relative SNR residual tolerated at the first coarsening step. Measured
#: ~8.9e-4 there; the bound leaves room for a noise-curve update without
#: leaving room for a lost factor.
FIRST_STEP_TOLERANCE = 5e-3


@pytest.fixture(scope="module")
def fine_catalog(mock_population: dict[str, np.ndarray]) -> Catalog:
    """The reference catalog every coarser grid is subsampled from."""
    return build_mock_catalog(
        mock_population,
        num_sources=NUM_SOURCES,
        f_min=F_MIN,
        f_max=F_MAX,
        df=FINE_DF,
    )


def test_subsampling_a_fine_catalog_matches_a_coarse_one(
    mock_population: dict[str, np.ndarray],
    fine_catalog: Catalog,
) -> None:
    """``[::k]`` of a fine catalog *is* the catalog built at ``k * df``.

    The premise of every other test in this module, and of
    ``notebooks/catalog_convergence.py``. Asserted as exact equality, not
    ``allclose``: both grids are formed from integer indices times a
    power-of-two ``df``, so anything less than bit-for-bit agreement means the
    grid construction changed.
    """
    fine_frequencies = np.asarray(fine_catalog.waveform_metadata.frequencies)
    fine_power = np.asarray(fine_catalog.polarization_power)

    for factor in SUBSAMPLE_FACTORS:
        coarse = build_mock_catalog(
            mock_population,
            num_sources=NUM_SOURCES,
            f_min=F_MIN,
            f_max=F_MAX,
            df=factor * FINE_DF,
        )
        np.testing.assert_array_equal(
            fine_frequencies[::factor],
            np.asarray(coarse.waveform_metadata.frequencies),
        )
        np.testing.assert_array_equal(
            fine_power[::factor], np.asarray(coarse.polarization_power)
        )


def _analysis_at(
    catalog: Catalog, factor: int, sensitivities: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-derive the masked analysis inputs on the grid coarsened by ``factor``.

    The PSD and the mask are rebuilt on the subsampled grid rather than
    subsampled themselves, so nothing about the coarse analysis is inherited
    from the fine one except the sources.
    """
    frequencies = jnp.asarray(catalog.waveform_metadata.frequencies)[::factor]
    polarization_power = jnp.asarray(catalog.polarization_power)[::factor]
    # The line this module exists to protect: df tracks the subsampling. Leave
    # it at the catalog's stored value and the SNR falls by exactly sqrt(k),
    # which is indistinguishable from convergence by eye.
    df = factor * FINE_DF

    network_psd = jnp.asarray(
        effective_psd(np.asarray(frequencies), DETECTORS, sensitivities)
    )
    mask = frequency_mask(frequencies, fmin=F_MIN, fmax=F_MAX) & jnp.isfinite(
        network_psd
    )
    frequencies, polarization_power, network_psd = apply_frequency_mask(
        mask, frequencies, polarization_power, network_psd
    )
    return {
        "df": df,
        "polarization_power": polarization_power,
        "effective_psd": network_psd,
        "num_bins": int(frequencies.shape[0]),
    }


@pytest.fixture(scope="module")
def resolutions(fine_catalog: Catalog) -> dict[int, dict[str, Any]]:
    """Masked analysis inputs, the injection, and SNR^2, at every resolution."""
    samples = catalog_samples(fine_catalog)
    redshift_grid = make_redshift_grid()
    total_merger_rate, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, samples, redshift_grid=redshift_grid
    )
    # The catalog is its own proposal: the density above and the reference
    # distance below are the same expressions the target forms at FIDUCIALS,
    # which is what makes every fiducial log-weight exactly zero.
    log_reference_distance = jnp.log(samples["luminosity_distance"]) + log_gw_em_ratio(
        samples["redshift"], FIDUCIALS["xi_0"], FIDUCIALS["xi_n"]
    )

    def weights_fn(params: dict[str, float]) -> tuple[jax.Array, jax.Array]:
        terms = bns_population(
            params, redshift_grid=redshift_grid
        ).compute_population_terms(samples)
        return terms.total_merger_rate, importance_log_weights(
            terms,
            proposal_log_prob=proposal_logprob,
            log_reference_distance=log_reference_distance,
        )

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
            average_mode="analytic_inclination",
        )
        run["snr_squared"] = float(
            spectral_snr_squared(
                run["observed"],
                run["effective_psd"],
                OBSERVATION_TIME * SECONDS_PER_YEAR,
                run["df"],
            )
        )
        runs[factor] = run
    return runs


def _log_likelihood(run: dict[str, Any], hubble_constant: float) -> float:
    """Gaussian log-density of the injection under the H0-shifted template."""
    noise_scale = gaussian_bin_scale(run["effective_psd"], OBSERVATION_TIME, run["df"])
    total_merger_rate, log_weights = run["weights_fn"](
        {**FIDUCIALS, "H0": hubble_constant}
    )
    model = spectral_density(
        run["polarization_power"],
        jnp.exp(log_weights),
        total_merger_rate,
        average_mode="analytic_inclination",
    )
    return float(jnp.sum(dist.Normal(model, noise_scale).log_prob(run["observed"])))


def test_snr_converges_under_frequency_refinement(
    resolutions: dict[int, dict[str, Any]],
) -> None:
    """Coarsening the grid costs SNR, monotonically and recoverably.

    The residual is signed and negative throughout: a coarse Riemann sum
    under-resolves the narrow low-frequency peak that carries the SNR, so it
    always *loses* signal rather than scattering about the truth.
    """
    reference = np.sqrt(resolutions[1]["snr_squared"])
    residuals = {
        factor: np.sqrt(resolutions[factor]["snr_squared"]) / reference - 1.0
        for factor in SUBSAMPLE_FACTORS
    }

    magnitudes = [abs(residuals[factor]) for factor in SUBSAMPLE_FACTORS]
    assert magnitudes == sorted(magnitudes), residuals
    assert all(residual < 0.0 for residual in residuals.values()), residuals
    assert abs(residuals[SUBSAMPLE_FACTORS[0]]) < FIRST_STEP_TOLERANCE, residuals
    # And the coarsest grid is genuinely bad, so the test would notice if the
    # whole sweep collapsed onto the reference.
    assert abs(residuals[SUBSAMPLE_FACTORS[-1]]) > 1e-2, residuals


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
        np.testing.assert_allclose(
            delta_residual, snr_squared_residual, rtol=1e-6, atol=0.0
        )


def test_every_log_weight_is_exactly_zero_at_the_fiducials(
    resolutions: dict[int, dict[str, Any]],
) -> None:
    """The premise the identity above rests on, stated on its own."""
    run = resolutions[1]
    _, log_weights = run["weights_fn"](dict(FIDUCIALS))
    assert isinstance(log_weights, jax.Array)
    np.testing.assert_array_equal(np.asarray(log_weights), 0.0)
