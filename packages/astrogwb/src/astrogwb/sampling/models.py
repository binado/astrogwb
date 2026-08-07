r"""NumPyro models for importance-weighted SGWB inference.

Two models share one amplitude marginalization. The inference model,
:func:`amplitude_marginalized_model`, integrates a multiplicative amplitude
direction out of the Gaussian likelihood and runs under NUTS; the
reconstruction model, :func:`amplitude_reconstruction_model`, is
*generative-only* and replays the chain's sufficient statistics through
:class:`~astrogwb.sampling.amplitude.AmplitudeConditional` under
:class:`~numpyro.infer.Predictive` to recover joint
:math:`(\varphi, \theta)` posterior draws.

They are two models, not one, for two reasons. First, the reconstruction must
never enter the inference potential: the amplitude direction is already
marginalized in the ``numpyro.factor`` site, so letting its sample site
contribute a ``log_prob`` would double-count it. Running it as a separate,
predictive-only model makes that impossible by construction. Second, the
reconstruction needs only the chain's three published statistics
(``amplitude_mle``, ``template_optimal_snr``, ``template_merger_rate``) plus
the prior and the scalings -- never the catalog or the :math:`(F, N)`
contraction -- so post-processing stays ``O(K)`` per draw and self-contained
against a saved chain.

End-to-end sketch (toy data; runnable as-is):

.. code-block:: python

    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro.distributions as dist
    import xarray as xr
    from numpyro.infer import MCMC, NUTS, Predictive

    from astrogwb.gwb import spectral_density
    from astrogwb.sampling import (
        amplitude_marginalized_model,
        amplitude_reconstruction_model,
        quadrature_grid,
    )

    # --- One-time setup: the prior and the grid the amplitude direction is
    # marginalized on. The scalings are absolute functions of H0; the model
    # anchors them at the fiducial itself.
    h0_fid = 70.0
    amplitude_prior = dist.Uniform(20.0, 140.0)
    grid = quadrature_grid(amplitude_prior, num_nodes=2001)

    def h0_merger_rate_amplitude(h0):
        return h0**-3

    def h0_amplitude(h0):
        return 1.0 / h0  # h0**-3 * h0**2

    frequencies = jnp.array([10.0, 30.0, 100.0])
    polarization_power = jnp.ones((3, 4))  # (F, N) toy catalog
    samples = {"redshift": jnp.linspace(0.1, 1.0, 4)}

    def toy_merger_rate_and_log_weights_fn(params, samples):
        rate = 10.0 ** params["log10_rate"] * (h0_fid / params["H0"]) ** 3
        return rate, jnp.zeros(samples["redshift"].shape[0])

    fiducials = {"H0": h0_fid, "log10_rate": -7.0}
    rate0, logw0 = toy_merger_rate_and_log_weights_fn(fiducials, samples)
    observed = spectral_density(
        polarization_power, jnp.exp(logw0), rate0,
        average_mode="analytic_inclination",
    )

    # --- Inference: NUTS on the amplitude-marginalized model.
    model = partial(
        amplitude_marginalized_model,
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=observed,
        effective_psd=jnp.ones_like(frequencies),
        observation_time=1.0,
        average_mode="analytic_inclination",
        merger_rate_and_log_weights_fn=toy_merger_rate_and_log_weights_fn,
        amplitude_parameter="H0",
        fiducials=fiducials,
        amplitude_fn=h0_amplitude,
        amplitude_prior=amplitude_prior,
        amplitude_grid=grid,
        priors={"log10_rate": dist.Uniform(-8.0, -6.0)},
    )
    mcmc = MCMC(NUTS(model), num_warmup=50, num_samples=50, num_chains=1,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))

    # --- Reconstruction: Predictive substitutes the chain's statistics into
    # the placeholder sites and draws H0 from AmplitudeConditional.
    draws = Predictive(
        partial(amplitude_reconstruction_model,
                amplitude_parameter="H0",
                amplitude_fn=h0_amplitude,
                merger_rate_amplitude_fn=h0_merger_rate_amplitude,
                prior=amplitude_prior, fiducial=h0_fid, grid=grid),
        posterior_samples=mcmc.get_samples(group_by_chain=True),
        batch_ndims=2,
        return_sites=["H0", "total_merger_rate", "quadrature_effective_nodes"],
    )(jax.random.fold_in(jax.random.PRNGKey(0), 1))

    # --- Merge: every returned site is (chain, draw); assign DataArrays.
    import arviz as az

    idata = az.from_numpyro(mcmc)
    for name, values in draws.items():
        idata.posterior[name] = xr.DataArray(
            np.asarray(values), dims=("chain", "draw")
        )
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

from astrogwb.detector import gaussian_bin_scale
from astrogwb.frequency import frequency_spacing, noise_weighted_inner_product
from astrogwb.gwb import (
    AverageMode,
    spectral_density,
)
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.importance.protocol import MergerRateAndLogWeightsFn
from astrogwb.sampling.amplitude import (
    AmplitudeConditional,
    AmplitudeFn,
    MergerRateAmplitudeFn,
)
from astrogwb.utils import years_to_seconds


def spectral_density_model(
    *,
    frequencies: jax.Array,
    polarization_power: jax.Array,
    samples: Mapping[str, jax.Array],
    observed_spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time: float,
    average_mode: AverageMode,
    merger_rate_and_log_weights_fn: MergerRateAndLogWeightsFn,
    priors: Mapping[str, dist.Distribution] | None = None,
    constants: Mapping[str, Any] | None = None,
) -> None:
    """NumPyro model for importance-weighted SGWB inference.

    Samples hyperparameters from ``priors``, evaluates a fixed proposal catalog
    through ``merger_rate_and_log_weights_fn``, and compares the predicted
    stochastic gravitational-wave background (SGWB) spectral density to
    ``observed_spectral_density`` under a per-frequency Gaussian likelihood.

    The predicted spectrum is built from precomputed per-source polarization
    powers and importance weights ``exp(log_weights)``; waveform generation is
    not part of this model. ``constants`` are merged with sampled parameters
    before the callback is invoked.

    This is the fully general model: every hyperparameter is sampled. See
    :func:`amplitude_marginalized_model` for the variant that integrates a
    multiplicative parameter out analytically.

    Registered sites:

    - one ``numpyro.sample`` per entry in ``priors``;
    - ``total_merger_rate`` and ``importance_relative_ess`` as deterministics;
    - ``spectral_density_obs`` as the observed Gaussian likelihood.

    Parameters
    ----------
    frequencies:
        Frequency grid in Hz, shape ``(F,)``. Must already be the analysis
        band; callers apply any frequency mask before invoking the model.
    polarization_power:
        Per-source polarization power at each frequency, shape ``(F, N)`` where
        ``N`` is the catalog size. Must share ``frequencies``.
    samples:
        Catalog arrays passed to ``merger_rate_and_log_weights_fn``. Each value
        should have leading dimension ``N``.
    observed_spectral_density:
        Observed SGWB spectral density at ``frequencies``, shape ``(F,)``.
    effective_psd:
        Network effective power spectral density at ``frequencies``, shape
        ``(F,)``.
    observation_time:
        Observation time in years, used only in the likelihood noise scale via
        :func:`astrogwb.detector.gaussian_bin_scale`.
    average_mode:
        How inclination is averaged when contracting polarization power:
        ``"analytic_inclination"`` applies the usual 0.4 factor;
        ``"catalog_inclination"`` uses the catalog weights directly.
    merger_rate_and_log_weights_fn:
        Callable ``(params, samples) -> (total_merger_rate, log_weights)``.
        ``params`` merges ``constants`` with sampled values; ``log_weights``
        has shape ``(N,)``. ``total_merger_rate`` is in mergers per second.
    priors:
        Mapping from parameter name to NumPyro prior distribution. Keys become
        sampled sites; defaults to an empty mapping (likelihood-only model).
    constants:
        Fixed parameter values merged into ``params`` for the callback. Defaults
        to an empty mapping.
    """
    sampled_params = {
        name: numpyro.sample(name, prior) for name, prior in (priors or {}).items()
    }
    params = {**(constants or {}), **sampled_params}

    total_merger_rate, log_weights = merger_rate_and_log_weights_fn(
        params,
        samples,
    )
    model_spectral_density = spectral_density(
        polarization_power,
        jnp.exp(log_weights),
        total_merger_rate,
        average_mode=average_mode,
    )

    numpyro.deterministic("total_merger_rate", total_merger_rate)
    numpyro.deterministic("importance_relative_ess", relative_ess(log_weights))

    scale = gaussian_bin_scale(effective_psd, frequencies, observation_time)
    numpyro.sample(
        "spectral_density_obs",
        dist.Normal(model_spectral_density, scale).to_event(1),
        obs=observed_spectral_density,
    )


def amplitude_marginalized_model(
    *,
    frequencies: jax.Array,
    polarization_power: jax.Array,
    samples: Mapping[str, jax.Array],
    observed_spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time: float,
    average_mode: AverageMode,
    merger_rate_and_log_weights_fn: MergerRateAndLogWeightsFn,
    amplitude_parameter: str,
    fiducials: Mapping[str, Any],
    amplitude_fn: AmplitudeFn,
    amplitude_prior: dist.Distribution,
    amplitude_grid: jax.Array | None = None,
    priors: Mapping[str, dist.Distribution] | None = None,
    constants: Mapping[str, Any] | None = None,
) -> None:
    r"""SGWB model with a multiplicative amplitude marginalized out of the likelihood.

    Identical to :func:`spectral_density_model` except that one strictly
    multiplicative parameter is integrated out instead of being sampled, which
    removes the long, curved amplitude--shape degeneracy that NUTS handles
    worst. The marginalization is essentially free here: the amplitude never
    touches the importance weights, and ``polarization_power`` is a fixed
    precomputed catalog.

    The physical parameter :math:`\varphi` is marginalized numerically under
    its own prior ``amplitude_prior``, for an arbitrary scaling
    :math:`f(\varphi)` to the multiplicative amplitude, with the integral
    evaluated by trapezoid quadrature on ``amplitude_grid``. What
    :meth:`~astrogwb.sampling.amplitude.AmplitudeConditional.sample` returns in
    post-processing is :math:`\varphi` itself (e.g. :math:`H_0`), not the
    amplitude. The only error is quadrature error, so grid resolution should be
    checked with
    :attr:`~astrogwb.sampling.amplitude.AmplitudeConditional.effective_nodes`.

    The callback is invoked with ``amplitude_parameter`` pinned to
    ``fiducials[amplitude_parameter]``, so the predicted spectrum it returns is
    the *template* :math:`\mathbf{m}(\theta)` and the marginalized amplitude
    :math:`A(\varphi) = f(\varphi)/f(\varphi_{\mathrm{fid}})` is the
    dimensionless ratio to that reference, with
    :math:`f = g_R \cdot g_F` factored into an independently scaling
    merger-rate piece and mean-energy-flux piece (see
    :func:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation.amplitude_scalings`).
    This covers a parameter entering directly, such as ``local_merger_rate``
    with :math:`g_R = \varphi`, :math:`g_F = 1`, and one entering inversely,
    such as :math:`H_0` with :math:`g_R = H_0^{-3}`, :math:`g_F = H_0^{2}`
    -- both work directly with the physical parameter name, no synthetic
    amplitude key required.

    Registered sites:

    - one ``numpyro.sample`` per entry in ``priors``;
    - ``template_merger_rate``, ``amplitude_mle``, ``template_optimal_snr``,
      and ``importance_relative_ess`` as deterministics;
    - ``amplitude_marginalized_log_likelihood`` as a ``numpyro.factor``.

    ``template_merger_rate`` is the rate at the pinned fiducial amplitude (the
    template), not the marginalized physical rate: the model never publishes
    a number that would be mistaken for the real merger rate at an
    unmarginalized :math:`\varphi`. Post-processing recovers the physical rate
    as ``template_merger_rate * g_R(varphi) / g_R(varphi_fid)``; see
    :func:`amplitude_reconstruction_model`.

    The two amplitude statistics are what post-processing needs to reconstruct
    joint :math:`(\varphi, \theta)` samples via
    :func:`amplitude_reconstruction_model` -- ``factor`` sites do not appear in
    ArviZ's posterior group, so they must be carried explicitly. They are preferred over the raw inner products because
    they are better conditioned, directly interpretable
    (:math:`\sigma_A = 1/\rho`), and invertible by multiplication alone:
    :math:`(m|m) = \rho^2` and :math:`(d|m) = \hat{A}\rho^2`. The contraction
    is deliberately :math:`\sigma`-space, :math:`\sum_i x_i y_i/\sigma_i^2`,
    and distinct from :func:`astrogwb.frequency.noise_weighted_inner_product`:
    routing it through the PSD-space function would make :math:`\rho^2` too
    small by :math:`2T`.

    Parameters
    ----------
    amplitude_parameter:
        Name of the parameter to marginalize over. Must be understood by
        ``merger_rate_and_log_weights_fn`` and must not appear in ``priors``.
    fiducials:
        Fiducial hyperparameters; ``fiducials[amplitude_parameter]`` is the
        reference value that defines the template.
    amplitude_fn:
        The absolute scaling :math:`f(\varphi) = g_R(\varphi) g_F(\varphi)`;
        the model anchors it at ``fiducials[amplitude_parameter]`` itself. Must
        be hashable by value -- see
        :class:`~astrogwb.sampling.amplitude.AmplitudeFn`.
    amplitude_prior:
        Prior :math:`\pi(\varphi)` on the marginalized parameter.
    amplitude_grid:
        Quadrature nodes for the marginalization integral. Defaults to
        :func:`~astrogwb.sampling.amplitude.quadrature_grid` of
        ``amplitude_prior``.

    Other parameters are as in :func:`spectral_density_model`.

    Raises
    ------
    ValueError
        If ``amplitude_parameter`` also appears in ``priors``. Sampling and
        marginalizing the same parameter is a silent double-counting with no
        visible symptom.
    """
    priors = priors or {}
    if amplitude_parameter in priors:
        raise ValueError(
            f"{amplitude_parameter!r} is marginalized analytically and cannot "
            "also be sampled; remove it from priors"
        )

    sampled_params = {
        name: numpyro.sample(name, prior) for name, prior in priors.items()
    }
    params = {
        **(constants or {}),
        **sampled_params,
        amplitude_parameter: fiducials[amplitude_parameter],
    }

    total_merger_rate, log_weights = merger_rate_and_log_weights_fn(
        params,
        samples,
    )
    model_spectral_density = spectral_density(
        polarization_power,
        jnp.exp(log_weights),
        total_merger_rate,
        average_mode=average_mode,
    )

    observation_time_sec = years_to_seconds(observation_time)
    df = frequency_spacing(frequencies)
    # scale = effective_psd / sqrt(2 T_sec df), so
    # sum(x * y / scale**2) = 2 T_sec * (x|y).
    template_norm = (
        2.0
        * observation_time_sec
        * noise_weighted_inner_product(
            model_spectral_density, model_spectral_density, effective_psd, df
        )
    )
    data_template = (
        2.0
        * observation_time_sec
        * noise_weighted_inner_product(
            observed_spectral_density, model_spectral_density, effective_psd, df
        )
    )
    amplitude_mle = data_template / template_norm
    template_optimal_snr = jnp.sqrt(template_norm)
    numpyro.deterministic("template_merger_rate", total_merger_rate)
    numpyro.deterministic("amplitude_mle", amplitude_mle)
    numpyro.deterministic("template_optimal_snr", template_optimal_snr)
    numpyro.deterministic("importance_relative_ess", relative_ess(log_weights))

    scale = gaussian_bin_scale(effective_psd, frequencies, observation_time)
    conditional = AmplitudeConditional(
        amplitude_mle,
        template_optimal_snr,
        amplitude_fn=amplitude_fn,
        prior=amplitude_prior,
        fiducial=fiducials[amplitude_parameter],
        grid=amplitude_grid,
    )
    log_likelihood_at_mle = (
        dist.Normal(
            amplitude_mle[..., None] * model_spectral_density,
            scale,
        )
        .to_event(1)
        .log_prob(observed_spectral_density)
    )
    # The marginalization factor *is* the normalizing constant of the
    # conditional that post-processing later samples.
    numpyro.factor(
        "amplitude_marginalized_log_likelihood",
        log_likelihood_at_mle + conditional.log_normalizer,
    )


def amplitude_reconstruction_model(
    *,
    amplitude_parameter: str,
    amplitude_fn: AmplitudeFn,
    merger_rate_amplitude_fn: MergerRateAmplitudeFn,
    prior: dist.Distribution,
    fiducial: Any,
    grid: jax.Array | None = None,
) -> None:
    r"""Generative-only reconstruction of joint :math:`(\varphi, \theta)` posterior draws.

    Consumed via :class:`~numpyro.infer.Predictive` with the posterior samples
    of :func:`amplitude_marginalized_model`; see this module's docstring for
    the end-to-end sketch. ``Predictive`` substitutes the chain's statistics
    into the placeholder sites below, draws the marginalized parameter from
    :class:`~astrogwb.sampling.amplitude.AmplitudeConditional` with a fresh key
    per posterior sample, and computes the deterministics from the substituted
    values. There is no forward physics here -- no catalog, no :math:`(F, N)`
    contraction -- so the cost is ``O(K)`` per draw and the model runs against
    a saved chain alone.

    Registered sites:

    - placeholder ``sample`` sites ``amplitude_mle``, ``template_optimal_snr``,
      and ``template_merger_rate`` -- never actually sampled, see below;
    - ``amplitude_parameter`` as a ``sample`` site from
      :class:`~astrogwb.sampling.amplitude.AmplitudeConditional`;
    - ``total_merger_rate`` and ``quadrature_effective_nodes`` as
      deterministics, the same names and definitions the reconstruction has
      always published.

    Parameters
    ----------
    amplitude_parameter:
        Name of the marginalized parameter; becomes the sample-site name of
        the reconstructed draws (e.g. ``"H0"``).
    amplitude_fn, prior, fiducial, grid:
        Must be exactly what :func:`amplitude_marginalized_model` was given.
        Reconstruction is only exact against the density the chain's factor
        site actually integrated; a silently different one yields a wrong
        marginalized posterior with no visible symptom, because the sufficient
        statistics stay finite and plausible whatever conditional you pair
        them with.
    merger_rate_amplitude_fn:
        The absolute merger-rate scaling :math:`g_R(\varphi)` alone, used to
        turn ``template_merger_rate`` back into the physical rate.
    """
    # The placeholder distributions are never sampled from
    # (ImproperUniform.sample raises NotImplementedError): they exist so
    # Predictive's substitution has a target. A missing or renamed statistic
    # therefore fails loudly instead of silently drawing from an improper
    # prior.
    placeholder = dist.ImproperUniform(constraints.real, (), ())
    amplitude_mle = numpyro.sample("amplitude_mle", placeholder)
    template_optimal_snr = numpyro.sample("template_optimal_snr", placeholder)
    template_merger_rate = numpyro.sample("template_merger_rate", placeholder)

    conditional = AmplitudeConditional(
        amplitude_mle,
        template_optimal_snr,
        amplitude_fn=amplitude_fn,
        prior=prior,
        fiducial=fiducial,
        grid=grid,
    )
    phi = jnp.asarray(numpyro.sample(amplitude_parameter, conditional))
    numpyro.deterministic(
        "total_merger_rate",
        template_merger_rate
        * merger_rate_amplitude_fn(phi)
        / merger_rate_amplitude_fn(jnp.asarray(fiducial)),
    )
    # `quadrature_effective_nodes` keeps its name even though the quadrature
    # object is gone: it is written into the posterior group and read back by
    # the paper's `run_mcmc.save`, so renaming it would break existing NetCDFs.
    numpyro.deterministic("quadrature_effective_nodes", conditional.effective_nodes)
