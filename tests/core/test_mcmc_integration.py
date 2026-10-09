r"""End-to-end NUTS runs against a realistic, physically consistent mock catalog.

Why the posterior widths are predictable. The catalog serves as both the
injection and the importance-sampling proposal, so every log-weight is exactly
zero and the "observed" spectrum is the unweighted catalog contraction. With
only :math:`H_0` free, the predicted spectrum is
:math:`S_h(H_0) = A\,S_h(H_0^{\rm fid})` with :math:`A = H_0^{\rm fid}/H_0`
(because the amplitude transform is :math:`H_0^{\rm fid}/H_0`), so the log-likelihood is
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

import logging
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from astrogwb_mock_population import (
    F_MAX,
    F_MIN,
    FIDUCIALS,
    build_reference_catalog,
    build_reference_spectrum,
    mock_population,
)
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, init_to_value

from astrogwb.constants import SECONDS_PER_YEAR
from astrogwb.detector import (
    effective_psd,
    gaussian_bin_scale,
    load_sensitivity_map,
)
from astrogwb.distributions.amplitude import AmplitudeConditional, amplitude_prior
from astrogwb.distributions.redshift.base import RedshiftDistribution
from astrogwb.frequency import apply_frequency_mask, frequency_mask
from astrogwb.gwb import spectral_snr
from astrogwb.inference import (
    SpectralDensityFn,
    amplitude_H0_transform,
    gwb_amplitude_marginalized_model,
    gwb_spectral_density_model,
)
from astrogwb.populations._types import SourceModel
from astrogwb.simulators.polarization_power import (
    CatalogMetadata,
    PolarizationPowerData,
)

pytestmark = pytest.mark.integration

DETECTORS: tuple[str, ...] = ("E1", "E2", "E3")

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

H0_PRIOR = dist.Uniform(20.0, 140.0)
H0_TRANSFORM = amplitude_H0_transform(FIDUCIALS["H0"])
AMPLITUDE_PRIOR = amplitude_prior(H0_PRIOR, H0_TRANSFORM)
OMEGA_M_PRIOR = dist.Normal(0.3096, 0.006)


class AnalysisInputs(NamedTuple):
    """Everything a model needs, plus the diagnostics the assertions use."""

    observed_spectral_density: jax.Array
    effective_psd: jax.Array
    observation_time: float
    scale: jax.Array
    estimator: SpectralDensityFn
    frequencies: jax.Array
    total_merger_rate: jax.Array
    snr: float


_TARGET = mock_population()


def pinned_target(
    params: Mapping[str, ArrayLike],
) -> tuple[RedshiftDistribution, SourceModel]:
    """The target population, unsampled hyperparameters pinned at the fiducials."""
    return _TARGET({**FIDUCIALS, **params})


class MarginalizedResult(NamedTuple):
    """One shared marginalized chain and its reconstructed H0 draws."""

    posterior: dict
    reconstructed: dict


def _build_analysis_inputs(
    data: PolarizationPowerData,
    metadata: CatalogMetadata,
    *,
    target_snr: float = TARGET_SNR,
) -> AnalysisInputs:
    """Reproduce the standard setup block against an in-memory catalog.

    Bind the reference catalog to the target, inject its prediction at the
    fiducials, load the network effective PSD, and mask out-of-band and
    non-finite bins.
    """
    frequencies = jnp.asarray(data["frequencies"])

    # The reference catalog is its own proposal and the injection is its
    # prediction at the fiducials, so every log-weight there is exactly zero
    # and the template equals the data.
    full_estimator, _ = build_reference_spectrum(
        data, metadata, population=pinned_target
    )
    observed_spectral_density, extras = full_estimator(FIDUCIALS)
    total_merger_rate = jnp.asarray(extras["total_merger_rate"])

    sensitivities = load_sensitivity_map(DETECTORS)
    network_psd = jnp.asarray(effective_psd(frequencies, DETECTORS, sensitivities))
    # effective_psd is inf wherever no detector pair contributes, and
    # Normal(loc, inf).log_prob is -inf, which kills NUTS with no diagnostic.
    mask = frequency_mask(frequencies, fmin=F_MIN, fmax=F_MAX) & jnp.isfinite(
        network_psd
    )
    assert int(jnp.sum(mask)) >= 2, "no usable frequency bins in the analysis band"

    # SNR^2 = 2 T (S_h|S_h) and S_eff carries no T, so rho scales as sqrt(T):
    # solve for the T that lands the injection on the target. The arrays stay
    # on the catalog's full grid and the band is a mask: bin widths derive
    # from the whole axis, so compressing first would mis-size the bins at the
    # edge of any gap.
    reference_snr = float(
        spectral_snr(
            observed_spectral_density,
            network_psd,
            REFERENCE_OBSERVATION_TIME * SECONDS_PER_YEAR,
            frequencies,
            frequency_mask=mask,
        )
    )
    observation_time = REFERENCE_OBSERVATION_TIME * (target_snr / reference_snr) ** 2
    snr = float(
        spectral_snr(
            observed_spectral_density,
            network_psd,
            observation_time * SECONDS_PER_YEAR,
            frequencies,
            frequency_mask=mask,
        )
    )
    np.testing.assert_allclose(snr, target_snr, rtol=1e-6)
    scale = gaussian_bin_scale(network_psd, observation_time, frequencies)
    frequencies, observed_spectral_density, network_psd, scale = apply_frequency_mask(
        mask, frequencies, observed_spectral_density, network_psd, scale
    )

    # Built on the masked grid: the band only chooses where the spectrum is
    # predicted, and the catalog's sources and weights are untouched.
    estimator = build_reference_spectrum(
        data, metadata, frequencies=frequencies, population=pinned_target
    )[0]

    return AnalysisInputs(
        observed_spectral_density=observed_spectral_density,
        effective_psd=network_psd,
        observation_time=observation_time,
        scale=scale,
        estimator=estimator,
        frequencies=frequencies,
        total_merger_rate=total_merger_rate,
        snr=snr,
    )


def _model_kwargs(inputs: AnalysisInputs) -> dict[str, object]:
    """Expose same-shaped data as dynamic arguments to NumPyro's JIT cache."""
    return {
        "observed_spectral_density": inputs.observed_spectral_density,
        "scale": inputs.scale,
    }


def _run_nuts(
    model,
    *,
    model_kwargs: dict[str, object],
    init_values: dict[str, float],
    seed: int = SEED,
) -> dict:
    """Run NUTS with model data passed dynamically through ``MCMC.run``."""
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
        jit_model_args=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), **model_kwargs)
    return mcmc.get_samples(group_by_chain=True)


def _direct_h0_model(inputs: AnalysisInputs, priors: dict[str, dist.Distribution]):
    return partial(
        gwb_spectral_density_model,
        spectral_density_fn=inputs.estimator,
        priors=priors,
    )


def _marginalized_model(
    inputs: AnalysisInputs,
    priors: dict[str, dist.Distribution],
):
    """H0 is a ``priors`` site pinned at its fiducial: the spectrum is the template."""
    return handlers.block(
        handlers.condition(
            partial(
                gwb_amplitude_marginalized_model,
                spectral_density_fn=inputs.estimator,
                amplitude_prior=AMPLITUDE_PRIOR,
                priors={"H0": H0_PRIOR, **priors},
            ),
            data={"H0": FIDUCIALS["H0"]},
        ),
        hide=["H0"],
    )


def _reconstruct_h0(posterior: dict) -> dict:
    """Draw H0 back from the chain's sufficient statistics."""
    conditional = AmplitudeConditional(
        posterior["amplitude_mle"],
        posterior["template_optimal_snr"],
        prior=AMPLITUDE_PRIOR,
    )
    # fold_in keeps the reconstruction key distinct from the chain's.
    amplitude = conditional.sample(jax.random.fold_in(jax.random.PRNGKey(SEED), 1))
    return {"H0": H0_TRANSFORM.inv(amplitude)}


@pytest.fixture(scope="module")
def analysis_inputs() -> AnalysisInputs:
    """Build the deterministic masked catalog inputs once for this module."""
    return _build_analysis_inputs(*build_reference_catalog())


@pytest.fixture(scope="module")
def marginalized_result(analysis_inputs: AnalysisInputs) -> MarginalizedResult:
    """Run the marginalized chain once for both tests that inspect it."""
    posterior = _run_nuts(
        _marginalized_model(analysis_inputs, {"Omega_m": OMEGA_M_PRIOR}),
        model_kwargs=_model_kwargs(analysis_inputs),
        init_values={"Omega_m": FIDUCIALS["Omega_m"]},
    )
    return MarginalizedResult(
        posterior=posterior,
        reconstructed=_reconstruct_h0(posterior),
    )


# --------------------------------------------------------------------------- #
# The direct model
# --------------------------------------------------------------------------- #
def test_h0_model_recovers_the_fiducial_and_the_fisher_width(
    analysis_inputs: AnalysisInputs,
) -> None:
    """NUTS on ``gwb_spectral_density_model`` lands on H0_fid with the Fisher width."""
    inputs = analysis_inputs

    posterior = _run_nuts(
        _direct_h0_model(inputs, {"H0": H0_PRIOR}),
        model_kwargs=_model_kwargs(inputs),
        init_values={"H0": FIDUCIALS["H0"]},
    )

    # Exactly these sites and no others: the source model runs isolated, so no
    # per-source (N,) column reaches the chain as a latent or deterministic.
    assert set(posterior) == {"H0", "total_merger_rate", "importance_relative_ess"}
    for name in ("H0", "total_merger_rate", "importance_relative_ess"):
        assert posterior[name].shape == (1, NUM_SAMPLES)

    # The catalog is its own proposal, so every weight is exactly 1: a far
    # sharper statement than the usual `> 0.1` collapse warning.
    np.testing.assert_allclose(
        np.asarray(posterior["importance_relative_ess"]), 1.0, rtol=1e-12
    )

    h0 = np.asarray(posterior["H0"])
    np.testing.assert_allclose(float(np.mean(h0)), FIDUCIALS["H0"], rtol=0.01)
    np.testing.assert_allclose(
        np.std(h0), FIDUCIALS["H0"] / inputs.snr, rtol=0.15, atol=0.0
    )


# --------------------------------------------------------------------------- #
# The amplitude-marginalized model
# --------------------------------------------------------------------------- #
def test_amplitude_marginalized_model_reconstructs_h0(
    analysis_inputs: AnalysisInputs,
    marginalized_result: MarginalizedResult,
) -> None:
    """H0 is marginalized out of the chain and drawn back to the same posterior."""
    inputs = analysis_inputs
    posterior = marginalized_result.posterior

    assert set(posterior) >= {
        "Omega_m",
        "total_merger_rate_at_unit_amplitude",
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
    np.testing.assert_allclose(
        float(np.mean(posterior["amplitude_mle"])), 1.0, rtol=5e-3
    )
    np.testing.assert_allclose(
        np.mean(posterior["template_optimal_snr"]),
        inputs.snr,
        rtol=5e-3,
        atol=0.0,
    )

    reconstructed = marginalized_result.reconstructed
    assert set(reconstructed) == {"H0"}
    for values in reconstructed.values():
        assert values.shape == (1, NUM_SAMPLES)

    h0 = np.asarray(reconstructed["H0"])
    np.testing.assert_allclose(float(np.mean(h0)), FIDUCIALS["H0"], rtol=0.01)
    np.testing.assert_allclose(
        np.std(h0), FIDUCIALS["H0"] / inputs.snr, rtol=0.15, atol=0.0
    )


def test_marginalized_and_direct_h0_posteriors_agree(
    analysis_inputs: AnalysisInputs,
    marginalized_result: MarginalizedResult,
) -> None:
    """The two models give the same H0 marginal on realistic data.

    ``test_spectral_inference.py::test_amplitude_marginalized_model_matches_the_general_model``
    already pins the *exact* log-density equivalence at ``rtol=1e-3`` by
    numerical quadrature, which is far sharper than any MCMC comparison. What
    this adds is coverage of the full pipeline -- NUTS, the
    reconstruction conditional -- on realistic data, which is where a
    mismatched conditional would actually bite.
    """
    inputs = analysis_inputs

    direct = _run_nuts(
        _direct_h0_model(inputs, {"H0": H0_PRIOR, "Omega_m": OMEGA_M_PRIOR}),
        model_kwargs=_model_kwargs(inputs),
        init_values={"H0": FIDUCIALS["H0"], "Omega_m": FIDUCIALS["Omega_m"]},
    )
    reconstructed = marginalized_result.reconstructed

    direct_h0 = np.asarray(direct["H0"])
    marginalized_h0 = np.asarray(reconstructed["H0"])
    # Both marginals are ~1.4 wide with an effective sample size of a few
    # hundred, so the Monte-Carlo error on each mean is ~0.1: agreement is
    # asserted at the level MCMC can support, not at the level the log-density
    # comparison already pins.
    np.testing.assert_allclose(
        float(np.mean(marginalized_h0)),
        float(np.mean(direct_h0)),
        rtol=0.0,
        atol=0.5,
    )
    np.testing.assert_allclose(
        np.std(marginalized_h0), np.std(direct_h0), rtol=0.2, atol=0.0
    )


# --------------------------------------------------------------------------- #
# One compiled sampler, several bands
# --------------------------------------------------------------------------- #
def _count_compilations(run) -> int:
    """XLA compilations triggered by ``run()``, counted off jax's own log."""
    records: list[str] = []

    class _Counter(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if message.startswith("Compiling"):
                records.append(message)

    logger = logging.getLogger("jax")
    handler = _Counter()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        with jax.log_compiles():
            run()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return len(records)


def test_one_compiled_sampler_serves_several_frequency_bands(
    analysis_inputs: AnalysisInputs,
) -> None:
    """Re-running on a sub-band must not pay for a second compilation.

    This is the whole reason the band is a traced mask rather than compressed
    arrays: compressing makes the bin count a *shape*, so every band is a new
    signature and a fresh sampler. Here one ``MCMC`` object is run twice with
    the same shapes and a different mask value, and the second run's
    compilations are counted.
    """
    inputs = analysis_inputs
    kwargs = _model_kwargs(inputs)
    midpoint = float(jnp.median(inputs.frequencies))
    wide = jnp.ones_like(inputs.frequencies, dtype=bool)
    narrow = inputs.frequencies <= midpoint
    assert 2 <= int(jnp.sum(narrow)) < int(jnp.sum(wide))

    mcmc = MCMC(
        NUTS(
            _direct_h0_model(inputs, {"H0": H0_PRIOR}),
            target_accept_prob=0.9,
            forward_mode_differentiation=True,
            init_strategy=init_to_value(values={"H0": FIDUCIALS["H0"]}),
        ),
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        progress_bar=False,
        jit_model_args=True,
    )

    def run(mask: jax.Array, seed: int):
        def go() -> None:
            mcmc.run(jax.random.PRNGKey(seed), **kwargs, frequency_mask=mask)

        compilations = _count_compilations(go)
        return compilations, mcmc.get_samples()["H0"]

    first, wide_h0 = run(wide, SEED)
    second, narrow_h0 = run(narrow, SEED)

    assert first > 0, "the first run compiled nothing: nothing was counted"
    # NumPyro rebuilds the sampler loop itself on every run; the model, the
    # spectrum and everything else in it must come from the first run's cache.
    assert second <= 1, (first, second)
    # A narrower band is less informative, so the two chains are genuinely
    # different -- the reused program is not one that ignores its mask.
    assert float(jnp.std(narrow_h0)) > float(jnp.std(wide_h0))
