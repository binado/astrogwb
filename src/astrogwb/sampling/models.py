r"""NumPyro inference from a spectrum callable and a prepared Gaussian scale.

``gwb_spectral_density_model`` accepts any ``SpectralDensityFn``. For example,
an analytic calculator can return no diagnostics::

    def analytic_spectrum(params):
        return params["amplitude"] * jnp.ones(3), {}

The importance-sampled spectrum is one too, once the catalog is bound to a
target outside inference::

    spectrum = build_importance_spectrum(
        catalog,
        source_model=target.source_model,
        merger_rate_fn=target.merger_rate_fn,
    )[0]
    model = partial(
        gwb_spectral_density_model,
        spectral_density_fn=spectrum,
        observed_spectral_density=observed,
        priors=priors,
        scale=gaussian_bin_scale(effective_psd, observation_time, df),
    )

For amplitude marginalization, diagnostics describe the pinned template, so an
importance caller relabels the rate without touching any other extras::

    template_spectrum = with_renamed_diagnostics(
        spectrum, {"total_merger_rate": "template_merger_rate"}
    )

``gwb_amplitude_marginalized_model`` publishes amplitude sufficient statistics.
``amplitude_reconstruction_model`` consumes those statistics and a template rate
via ``Predictive`` to recover joint amplitude/shape draws and the physical rate.
Keep reconstruction separate from inference to avoid counting amplitude twice.
When no rate is available, draw amplitudes directly from ``AmplitudeConditional``
using the same statistics, prior, fiducial, scaling function, and grid.

End-to-end sketch (toy data; runnable as-is):

.. code-block:: python

    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro.distributions as dist
    import xarray as xr
    from numpyro.infer import MCMC, NUTS, Predictive

    from astrogwb.distributions.amplitude import quadrature_grid
    from astrogwb.detector import gaussian_bin_scale
    from astrogwb.sampling import (
        gwb_amplitude_marginalized_model,
        amplitude_reconstruction_model,
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

    def toy_spectrum(params):
        template_rate = 10.0 ** params["log10_rate"] * (h0_fid / params["H0"]) ** 3
        prediction = 0.4 * template_rate * (params["H0"] / h0_fid) ** 2 * jnp.ones(3)
        return prediction, {"template_merger_rate": template_rate}

    fiducials = {"H0": h0_fid, "log10_rate": -7.0}
    observed, _ = toy_spectrum(fiducials)

    # --- Inference: NUTS on the amplitude-marginalized model.
    model = partial(
        gwb_amplitude_marginalized_model,
        spectral_density_fn=toy_spectrum,
        observed_spectral_density=observed,
        scale=gaussian_bin_scale(jnp.ones(3), 1.0, 0.25),
        amplitude_parameter="H0",
        amplitude_fiducial=h0_fid,
        amplitude_fn=h0_amplitude,
        amplitude_prior=amplitude_prior,
        amplitude_grid=grid,
        priors={"log10_rate": dist.Uniform(-8.0, -6.0)},
    )
    mcmc = MCMC(NUTS(model), num_warmup=50, num_samples=50, num_chains=1,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))

    # --- Reconstruction: pass the chain's statistics as a (chain, draw)
    # batch and draw one H0 for every element.
    chain_samples = mcmc.get_samples(group_by_chain=True)
    draws = Predictive(
        partial(amplitude_reconstruction_model,
                amplitude_parameter="H0",
                amplitude_fn=h0_amplitude,
                merger_rate_amplitude_fn=h0_merger_rate_amplitude,
                prior=amplitude_prior, fiducial=h0_fid, grid=grid),
        num_samples=1,
        return_sites=["H0", "total_merger_rate", "quadrature_effective_nodes"],
    )(
        jax.random.fold_in(jax.random.PRNGKey(0), 1),
        amplitude_mle=chain_samples["amplitude_mle"],
        template_optimal_snr=chain_samples["template_optimal_snr"],
        template_merger_rate=chain_samples["template_merger_rate"],
    )
    draws = {name: values[0] for name, values in draws.items()}

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
from jax.typing import ArrayLike

from astrogwb.distributions.amplitude import (
    AmplitudeConditional,
    AmplitudeFn,
    MergerRateAmplitudeFn,
)
from astrogwb.sampling.protocol import SpectralDensityFn


def gwb_spectral_density_model(
    *,
    spectral_density_fn: SpectralDensityFn,
    observed_spectral_density: jax.Array,
    priors: Mapping[str, dist.Distribution],
    scale: jax.Array,
    frequency_mask: jax.Array | None = None,
) -> None:
    """Sample parameters and compare a supplied spectrum to Gaussian observations.

    ``spectral_density_fn(params)`` returns a spectrum of shape ``(F,)`` and
    optional deterministic diagnostics. ``scale`` is the prepared per-bin
    standard deviation, normally ``gaussian_bin_scale(psd, time_years, df)``.
    Use ``priors={}`` for likelihood-only evaluation. The observation site is
    ``spectral_density_obs`` with one frequency event dimension. Diagnostic
    names must not collide with priors or that observation site.

    ``frequency_mask`` is an optional boolean array of shape ``(F,)`` selecting
    the bins the likelihood counts. It is a *traced* argument on a fixed grid,
    so sweeping its value never triggers a recompile -- only changing its
    length does, since that changes every array's shape. Excluded bins
    contribute exactly zero, so the result equals evaluating the model on the
    arrays compressed to the selection, and a masked bin's ``scale`` may be
    infinite without producing a non-finite log density or gradient.

    The mask is applied to the observation *site*
    (``dist.Normal(...).mask(...)``) rather than with
    :func:`numpyro.handlers.mask`: a handler applies to every sample site in
    its scope, so an ``(F,)`` mask would also reach the scalar prior sites and
    broadcast their log densities to shape ``(F,)``.
    """
    params = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
    prediction, extras = spectral_density_fn(params)
    for name, value in extras.items():
        numpyro.deterministic(name, value)
    observation = dist.Normal(prediction, scale)
    if frequency_mask is not None:
        observation = observation.mask(frequency_mask)
    numpyro.sample(
        "spectral_density_obs",
        observation.to_event(1),
        obs=observed_spectral_density,
    )


def gwb_amplitude_marginalized_model(
    *,
    spectral_density_fn: SpectralDensityFn,
    observed_spectral_density: jax.Array,
    priors: Mapping[str, dist.Distribution],
    scale: jax.Array,
    amplitude_parameter: str,
    amplitude_fiducial: ArrayLike,
    amplitude_fn: AmplitudeFn,
    amplitude_prior: dist.Distribution,
    amplitude_grid: jax.Array | None = None,
    frequency_mask: jax.Array | None = None,
) -> None:
    """Marginalize a multiplicative parameter of any supplied spectrum.

    Sample shape parameters from ``priors`` and evaluate the spectrum once with
    ``amplitude_parameter`` pinned to ``amplitude_fiducial``. The amplitude
    ratio is ``amplitude_fn(phi) / amplitude_fn(amplitude_fiducial)``; the
    spectrum must factor this way throughout the amplitude prior's support.
    Integrate under ``amplitude_prior`` using ``AmplitudeConditional`` and
    its default quadrature grid when ``amplitude_grid`` is omitted.

    Records ``amplitude_mle``, ``template_optimal_snr``, the returned extras,
    and the ``amplitude_marginalized_log_likelihood`` factor. Diagnostics must
    not collide with priors or these three likelihood-owned names. No merger
    rate is required and extras are never rescaled. Importance callers should
    rename their spectrum's ``total_merger_rate`` to ``template_merger_rate``
    before returning it; see the module example. Use the unchanged
    ``amplitude_reconstruction_model`` for rate-aware reconstruction, or
    ``AmplitudeConditional`` directly when only amplitude draws are needed.

    ``frequency_mask`` is an optional boolean array of shape ``(F,)`` selecting
    the bins the likelihood counts, as in :func:`gwb_spectral_density_model`.
    Every sum below restricts to it.

    Raises ``ValueError`` if the amplitude is also present in ``priors``.
    All spectrum, observation, and scale arrays have shape ``(F,)``.
    """
    if amplitude_parameter in priors:
        raise ValueError(
            f"{amplitude_parameter!r} is marginalized analytically and cannot "
            "also be sampled; remove it from priors"
        )
    params = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
    params[amplitude_parameter] = amplitude_fiducial
    model_spectral_density, extras = spectral_density_fn(params)
    for name, value in extras.items():
        numpyro.deterministic(name, value)

    inverse_variance = scale**-2
    # `log_scale` is the per-bin Gaussian normalization, summed further down.
    # Masking both arrays here is what restricts every sum below: an excluded
    # bin contributes zero inverse variance to the three sufficient statistics
    # and zero to the normalization, which is exactly dropping it.
    log_scale = jnp.log(scale) + 0.5 * jnp.log(2.0 * jnp.pi)
    if frequency_mask is not None:
        keep = jnp.asarray(frequency_mask)
        inverse_variance = jnp.where(keep, inverse_variance, 0.0)
        log_scale = jnp.where(keep, log_scale, 0.0)
    template_norm = jnp.sum(
        model_spectral_density * model_spectral_density * inverse_variance
    )
    data_template = jnp.sum(
        observed_spectral_density * model_spectral_density * inverse_variance
    )
    amplitude_mle = data_template / template_norm
    template_optimal_snr = jnp.sqrt(template_norm)
    numpyro.deterministic("amplitude_mle", amplitude_mle)
    numpyro.deterministic("template_optimal_snr", template_optimal_snr)

    conditional = AmplitudeConditional(
        amplitude_mle,
        template_optimal_snr,
        amplitude_fn=amplitude_fn,
        prior=amplitude_prior,
        fiducial=amplitude_fiducial,
        grid=amplitude_grid,
    )
    # log p(d | A_mle) = -1/2 chi^2(A_mle, theta) + normalization, with
    # chi^2(A_mle, theta) = data_norm - A_mle * data_template (the completed
    # square, eq. rearranged-amplitude-joint-likelihood in the paper). This
    # reuses amplitude_mle and data_template instead of re-forming the (F,)
    # residual d - A_mle * m through a second Normal.log_prob evaluation;
    # data_norm depends only on fixed inputs, never on theta or phi.
    data_norm = jnp.sum(
        observed_spectral_density * observed_spectral_density * inverse_variance
    )
    normalization = -jnp.sum(log_scale)
    log_likelihood_at_mle = normalization - 0.5 * (
        data_norm - amplitude_mle * data_template
    )
    # The marginalization factor *is* the normalizing constant of the
    # conditional that post-processing later samples.
    numpyro.factor(
        "amplitude_marginalized_log_likelihood",
        log_likelihood_at_mle + conditional.log_normalizer,
    )


def amplitude_reconstruction_model(
    amplitude_mle: jax.Array,
    template_optimal_snr: jax.Array,
    template_merger_rate: jax.Array,
    *,
    amplitude_parameter: str,
    amplitude_fn: AmplitudeFn,
    merger_rate_amplitude_fn: MergerRateAmplitudeFn,
    prior: dist.Distribution,
    fiducial: Any,
    grid: jax.Array | None = None,
) -> None:
    r"""Generative-only reconstruction of joint :math:`(\varphi, \theta)` posterior draws.

    Consumed via :class:`~numpyro.infer.Predictive` with the sufficient
    statistics published by :func:`gwb_amplitude_marginalized_model`; see this
    module's docstring for the end-to-end sketch. The statistics' broadcast
    shape is the batch shape of
    :class:`~astrogwb.distributions.amplitude.AmplitudeConditional`, so a
    ``(chain, draw)`` input batch yields one independent marginalized-parameter
    draw for every chain element. There is no forward physics here -- no
    catalog, no :math:`(F, N)` contraction -- so the cost is ``O(K)`` per draw
    and the model runs against a saved chain alone.

    Registered sites:

    - ``amplitude_parameter`` as a ``sample`` site from
      :class:`~astrogwb.distributions.amplitude.AmplitudeConditional`;
    - ``total_merger_rate`` and ``quadrature_effective_nodes`` as
      deterministics, the same names and definitions the reconstruction has
      always published.

    Parameters
    ----------
    amplitude_mle, template_optimal_snr, template_merger_rate:
        Sufficient statistics published by
        :func:`gwb_amplitude_marginalized_model`. Their shapes must broadcast to a
        common batch shape; production reconstruction passes ``(chain, draw)``
        arrays.
    amplitude_parameter:
        Name of the marginalized parameter; becomes the sample-site name of
        the reconstructed draws (e.g. ``"H0"``).
    amplitude_fn, prior, fiducial, grid:
        Must be exactly what :func:`gwb_amplitude_marginalized_model` was given.
        Reconstruction is only exact against the density the chain's factor
        site actually integrated; a silently different one yields a wrong
        marginalized posterior with no visible symptom, because the sufficient
        statistics stay finite and plausible whatever conditional you pair
        them with.
    merger_rate_amplitude_fn:
        The absolute merger-rate scaling :math:`g_R(\varphi)` alone, used to
        turn ``template_merger_rate`` back into the physical rate.
    """
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
