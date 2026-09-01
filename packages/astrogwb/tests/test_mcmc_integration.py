r"""End-to-end NUTS runs against a realistic, physically consistent mock catalog.

These tests reproduce ``examples/h0_mcmc.py`` and
``examples/amplitude_marginalized_model.py`` at small sampler settings, on a
catalog built in-process from the committed population fixture (see
``conftest.mock_catalog_factory``). Unlike the paper package's
``test_inference_pipeline.py``, which drives the same models on
``rng.uniform(0, 1)`` power and ``d_L = 1e3 (1 + z)``, the data here carries
real physics: a wrong factor of :math:`(1 + z)`, a mis-normalized mass moment,
or a broken amplitude reconstruction changes the numbers these tests assert.

Everything is deterministic. The seed is fixed, the fixture is frozen, and the
injection carries **no noise realization** -- a draw would shift the posterior
by ~1 sigma and make a correct model look broken. So these tests do not flake:
they either pass or report a real change.

Why the posterior widths are predictable. The catalog serves as both the
injection and the importance-sampling proposal, so every log-weight is exactly
zero and the "observed" spectrum is the unweighted catalog contraction. With
only :math:`H_0` free, the predicted spectrum is
:math:`S_h(H_0) = A\,S_h(H_0^{\rm fid})` with :math:`A = H_0^{\rm fid}/H_0`
(because ``amplitude_H0_fn`` is :math:`H_0^{-1}`), so the log-likelihood is
exactly :math:`-\tfrac{1}{2}(1 - A)^2\rho^2`. Hence :math:`A \sim N(1,
1/\rho^2)` truncated by the prior, giving :math:`\sigma_{H_0}/H_0^{\rm fid} =
1/\rho` and a mean biased high by only :math:`1/\rho^2` (~4e-4 at
:math:`\rho = 50`) -- negligible against the Monte-Carlo standard error of the
chain.

The observation time is *derived*, not fixed: it is solved for the target SNR
from a reference run, using :math:`\rho \propto \sqrt{T}`. If a bundled noise
curve or the analysis band ever changes, these tests stay in the intended
linear regime instead of silently drifting out of it.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
import xarray as xr
from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import effective_psd, load_sensitivity_map
from astrogwb.frequency import apply_frequency_mask, frequency_mask
from astrogwb.gwb import spectral_density, spectral_snr
from astrogwb.importance.models.bns_madau_dickinson_modified_propagation import (
    amplitude_H0_fn,
    compute_merger_rate_distance_and_logprob,
    make_merger_rate_and_log_weights_fn,
    merger_rate_H0_fn,
)
from astrogwb.importance.protocol import MergerRateAndLogWeightsFn
from astrogwb.sampling import (
    amplitude_marginalized_model,
    amplitude_reconstruction_model,
    quadrature_grid,
)
from astrogwb.sampling.models import spectral_density_model
from astrogwb_mock_population import FIDUCIALS, make_redshift_grid
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value

pytestmark = pytest.mark.integration

DETECTORS: tuple[str, ...] = ("E1", "E2", "E3")
F_MIN = 10.0
F_MAX = 512.0

#: Target network SNR of the injection. Chosen so ``sigma_H0/H0 == 1/SNR`` is
#: tested in the clean linear regime, well inside the ``Uniform(20, 140)``
#: prior: at rho = 50 the posterior is ~1.35 wide against a 120-wide prior.
TARGET_SNR = 50.0

#: Reference observation time, in years, the SNR solve is anchored at.
REFERENCE_OBSERVATION_TIME = 1.0

SEED = 42
#: After XLA compilation a NUTS draw on one or two latents is essentially
#: free -- 1000 draws cost the same wall time as 200 -- so the count is set by
#: the Monte-Carlo error the assertions can tolerate, not by runtime. At 200
#: draws the effective sample size is ~50 and the MCSE on a posterior standard
#: deviation is ~10%, too coarse to pin the Fisher width; at 1000 it is ~380.
NUM_WARMUP = 500
NUM_SAMPLES = 1000

#: Nodes in the H0 quadrature grid the marginalization runs on. Smaller than
#: the examples' 1024; ``quadrature_effective_nodes`` is asserted rather than
#: assumed.
AMPLITUDE_NUM_NODES = 512

H0_PRIOR = dist.Uniform(20.0, 140.0)
OMEGA_M_PRIOR = dist.Normal(0.3096, 0.006)


class AnalysisInputs(NamedTuple):
    """Everything a model needs, plus the diagnostics the assertions use."""

    polarization_power: jax.Array
    samples: dict[str, jax.Array]
    observed_spectral_density: jax.Array
    effective_psd: jax.Array
    observation_time: float
    df: float
    merger_rate_and_log_weights_fn: MergerRateAndLogWeightsFn
    frequencies: jax.Array
    total_merger_rate: jax.Array
    snr: float


def _build_analysis_inputs(
    catalog: xr.Dataset,
    *,
    target_snr: float = TARGET_SNR,
) -> AnalysisInputs:
    """Reproduce the examples' setup block against an in-memory catalog.

    Mirrors ``examples/h0_mcmc.py``: unpack the catalog, build the redshift
    grid, call :func:`compute_merger_rate_distance_and_logprob` *once* for both
    the injection rate and the proposal log-density, contract with unit
    weights, load the network effective PSD, and mask out-of-band and
    non-finite bins.
    """
    frequencies = jnp.asarray(catalog.frequency.values)
    df = float(catalog.attrs["df"])
    polarization_power = jnp.asarray(catalog.polarization_power.values)
    samples = {
        str(name): jnp.asarray(catalog.source_parameters.sel(parameter=name).values)
        for name in catalog.parameter.values
    }
    num_sources = polarization_power.shape[1]
    redshift_grid = make_redshift_grid()

    # One call yields both the fiducial rate (for the injection) and the
    # proposal log-density (for the weights). Sharing it is what makes every
    # weight exactly 1 at the fiducials.
    total_merger_rate, _, proposal_logprob = compute_merger_rate_distance_and_logprob(
        FIDUCIALS, samples, redshift_grid=redshift_grid
    )
    observed_spectral_density = spectral_density(
        polarization_power,
        jnp.ones(num_sources),
        total_merger_rate,
        average_mode="analytic_inclination",
    )

    weights_fn = make_merger_rate_and_log_weights_fn(
        fiducials=FIDUCIALS,
        redshift_grid=redshift_grid,
        proposal_logprob=proposal_logprob,
    )

    def merger_rate_and_log_weights(params, samples):
        """Take every hyperparameter the chain does not sample from the fiducials."""
        return weights_fn({**FIDUCIALS, **params}, samples)

    sensitivities = load_sensitivity_map(DETECTORS)
    network_psd = jnp.asarray(effective_psd(frequencies, DETECTORS, sensitivities))
    # effective_psd is inf wherever no detector pair contributes, and
    # Normal(loc, inf).log_prob is -inf, which kills NUTS with no diagnostic.
    mask = frequency_mask(frequencies, fmin=F_MIN, fmax=F_MAX) & jnp.isfinite(
        network_psd
    )
    assert int(jnp.sum(mask)) >= 2, "no usable frequency bins in the analysis band"
    frequencies, polarization_power, observed_spectral_density, network_psd = (
        apply_frequency_mask(
            mask,
            frequencies,
            polarization_power,
            observed_spectral_density,
            network_psd,
        )
    )

    # SNR^2 = 2 T (S_h|S_h) and S_eff carries no T, so rho scales as sqrt(T):
    # solve for the T that lands the injection on the target.
    reference_snr = float(
        spectral_snr(
            observed_spectral_density,
            network_psd,
            REFERENCE_OBSERVATION_TIME * SECONDS_PER_YEAR,
            df,
        )
    )
    observation_time = REFERENCE_OBSERVATION_TIME * (target_snr / reference_snr) ** 2
    snr = float(
        spectral_snr(
            observed_spectral_density,
            network_psd,
            observation_time * SECONDS_PER_YEAR,
            df,
        )
    )
    np.testing.assert_allclose(snr, target_snr, rtol=1e-12, atol=0.0)

    return AnalysisInputs(
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=observed_spectral_density,
        effective_psd=network_psd,
        observation_time=observation_time,
        df=df,
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights,
        frequencies=frequencies,
        total_merger_rate=total_merger_rate,
        snr=snr,
    )


def _run_nuts(model, *, init_values: dict[str, float], seed: int = SEED) -> dict:
    """Run the shared NUTS configuration and return grouped ``(chain, draw)`` draws."""
    # One or two latents against an (F, N) matvec is exactly the case
    # forward-mode AD is built for: reverse mode would tape the contraction.
    kernel = NUTS(
        model,
        target_accept_prob=0.9,
        forward_mode_differentiation=True,
        init_strategy=init_to_value(values=init_values),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc.get_samples(group_by_chain=True)


def _direct_h0_model(inputs: AnalysisInputs, priors: dict[str, dist.Distribution]):
    return partial(
        spectral_density_model,
        polarization_power=inputs.polarization_power,
        samples=inputs.samples,
        observed_spectral_density=inputs.observed_spectral_density,
        effective_psd=inputs.effective_psd,
        observation_time=inputs.observation_time,
        df=inputs.df,
        average_mode="analytic_inclination",
        merger_rate_and_log_weights_fn=inputs.merger_rate_and_log_weights_fn,
        priors=priors,
    )


def _marginalized_model(
    inputs: AnalysisInputs,
    priors: dict[str, dist.Distribution],
    amplitude_grid: jax.Array,
):
    return partial(
        amplitude_marginalized_model,
        polarization_power=inputs.polarization_power,
        samples=inputs.samples,
        observed_spectral_density=inputs.observed_spectral_density,
        effective_psd=inputs.effective_psd,
        observation_time=inputs.observation_time,
        df=inputs.df,
        average_mode="analytic_inclination",
        merger_rate_and_log_weights_fn=inputs.merger_rate_and_log_weights_fn,
        amplitude_parameter="H0",
        fiducials=FIDUCIALS,
        # Passed by name, never wrapped: AmplitudeConditional hashes
        # amplitude_fn into the jit cache key, so a freshly-minted callable
        # retraces the model on every construction. Pinned by
        # test_amplitude_scalings.py.
        amplitude_fn=amplitude_H0_fn,
        amplitude_prior=H0_PRIOR,
        amplitude_grid=amplitude_grid,
        priors=priors,
    )


def _reconstruct_h0(posterior: dict, amplitude_grid: jax.Array) -> dict:
    """Draw H0 back from the chain's sufficient statistics."""
    draws = Predictive(
        partial(
            amplitude_reconstruction_model,
            amplitude_parameter="H0",
            amplitude_fn=amplitude_H0_fn,
            merger_rate_amplitude_fn=merger_rate_H0_fn,
            prior=H0_PRIOR,
            fiducial=FIDUCIALS["H0"],
            grid=amplitude_grid,
        ),
        num_samples=1,
        return_sites=["H0", "total_merger_rate", "quadrature_effective_nodes"],
    )(
        # fold_in keeps the reconstruction key distinct from the chain's.
        jax.random.fold_in(jax.random.PRNGKey(SEED), 1),
        amplitude_mle=posterior["amplitude_mle"],
        template_optimal_snr=posterior["template_optimal_snr"],
        template_merger_rate=posterior["template_merger_rate"],
    )
    return {name: values[0] for name, values in draws.items()}


# --------------------------------------------------------------------------- #
# The direct model
# --------------------------------------------------------------------------- #
def test_h0_model_recovers_the_fiducial_and_the_fisher_width(
    mock_catalog_factory,
) -> None:
    """NUTS on ``spectral_density_model`` lands on H0_fid with the Fisher width."""
    inputs = _build_analysis_inputs(mock_catalog_factory())

    posterior = _run_nuts(
        _direct_h0_model(inputs, {"H0": H0_PRIOR}),
        init_values={"H0": FIDUCIALS["H0"]},
    )

    assert set(posterior) >= {"H0", "total_merger_rate", "importance_relative_ess"}
    for name in ("H0", "total_merger_rate", "importance_relative_ess"):
        assert posterior[name].shape == (1, NUM_SAMPLES)

    # The catalog is its own proposal, so every weight is exactly 1. This is a
    # far sharper statement than the examples' `> 0.1` collapse warning.
    np.testing.assert_allclose(
        np.asarray(posterior["importance_relative_ess"]), 1.0, rtol=1e-12
    )

    h0 = np.asarray(posterior["H0"])
    assert float(np.mean(h0)) == pytest.approx(FIDUCIALS["H0"], rel=0.01)
    np.testing.assert_allclose(
        np.std(h0), FIDUCIALS["H0"] / inputs.snr, rtol=0.15, atol=0.0
    )


# --------------------------------------------------------------------------- #
# The amplitude-marginalized model
# --------------------------------------------------------------------------- #
def test_amplitude_marginalized_model_reconstructs_h0(mock_catalog_factory) -> None:
    """H0 is marginalized out of the chain and drawn back to the same posterior."""
    inputs = _build_analysis_inputs(mock_catalog_factory())
    # Built once and shared by the chain and the reconstruction: the
    # reconstruction is only exact against the very density the factor site
    # integrated, and a separately-constructed prior or grid is exactly the
    # mismatch that goes unnoticed.
    amplitude_grid = quadrature_grid(H0_PRIOR, num_nodes=AMPLITUDE_NUM_NODES)

    posterior = _run_nuts(
        _marginalized_model(inputs, {"Omega_m": OMEGA_M_PRIOR}, amplitude_grid),
        init_values={"Omega_m": FIDUCIALS["Omega_m"]},
    )

    assert set(posterior) >= {
        "Omega_m",
        "template_merger_rate",
        "amplitude_mle",
        "template_optimal_snr",
        "importance_relative_ess",
    }
    # A numpyro.factor publishes no draws, so neither the marginalized
    # parameter nor the likelihood site can appear in the chain.
    assert "H0" not in posterior
    assert "spectral_density_obs" not in posterior

    # The template equals the data at the fiducial, so the best-fit amplitude
    # ratio averages to 1 and the template's optimal SNR to the injection's.
    # Per draw they scatter by ~1%: Omega_m is sampled, so an individual
    # template is not the injection.
    assert float(np.mean(posterior["amplitude_mle"])) == pytest.approx(1.0, rel=5e-3)
    np.testing.assert_allclose(
        np.mean(posterior["template_optimal_snr"]),
        inputs.snr,
        rtol=5e-3,
        atol=0.0,
    )

    reconstructed = _reconstruct_h0(posterior, amplitude_grid)
    assert set(reconstructed) == {
        "H0",
        "total_merger_rate",
        "quadrature_effective_nodes",
    }
    for values in reconstructed.values():
        assert values.shape == (1, NUM_SAMPLES)

    h0 = np.asarray(reconstructed["H0"])
    assert float(np.mean(h0)) == pytest.approx(FIDUCIALS["H0"], rel=0.01)
    np.testing.assert_allclose(
        np.std(h0), FIDUCIALS["H0"] / inputs.snr, rtol=0.15, atol=0.0
    )

    # A health check on the amplitude grid: the conditional is ~1/rho wide over
    # a 120-wide prior, so a grid too coarse to resolve it collapses to a
    # handful of nodes.
    assert np.all(np.asarray(reconstructed["quadrature_effective_nodes"]) > 10.0)


def test_marginalized_and_direct_h0_posteriors_agree(mock_catalog_factory) -> None:
    """The two models give the same H0 marginal on realistic data.

    ``test_sampling.py::test_amplitude_marginalized_model_matches_the_general_model``
    already pins the *exact* log-density equivalence at ``rtol=1e-3`` by
    numerical quadrature, which is far sharper than any MCMC comparison. What
    this adds is coverage of the full pipeline -- NUTS, ``Predictive``, the
    reconstruction conditional -- on realistic data, which is where a
    mismatched conditional would actually bite.
    """
    inputs = _build_analysis_inputs(mock_catalog_factory())
    amplitude_grid = quadrature_grid(H0_PRIOR, num_nodes=AMPLITUDE_NUM_NODES)

    direct = _run_nuts(
        _direct_h0_model(inputs, {"H0": H0_PRIOR, "Omega_m": OMEGA_M_PRIOR}),
        init_values={"H0": FIDUCIALS["H0"], "Omega_m": FIDUCIALS["Omega_m"]},
    )
    marginalized = _run_nuts(
        _marginalized_model(inputs, {"Omega_m": OMEGA_M_PRIOR}, amplitude_grid),
        init_values={"Omega_m": FIDUCIALS["Omega_m"]},
    )
    reconstructed = _reconstruct_h0(marginalized, amplitude_grid)

    direct_h0 = np.asarray(direct["H0"])
    marginalized_h0 = np.asarray(reconstructed["H0"])
    # Both marginals are ~1.4 wide with an effective sample size of a few
    # hundred, so the Monte-Carlo error on each mean is ~0.1: agreement is
    # asserted at the level MCMC can support, not at the level the log-density
    # comparison already pins.
    assert float(np.mean(marginalized_h0)) == pytest.approx(
        float(np.mean(direct_h0)), abs=0.5
    )
    np.testing.assert_allclose(
        np.std(marginalized_h0), np.std(direct_h0), rtol=0.2, atol=0.0
    )
