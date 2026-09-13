r"""NumPyro inference from a spectrum callable and a prepared Gaussian scale.

``gwb_spectral_density_model`` accepts any ``SpectralDensityFn``. For example,
an analytic calculator can return no diagnostics::

    def analytic_spectrum(params):
        return params["amplitude"] * jnp.ones(3), {}

The importance-sampled spectrum is one too, once the catalog is bound to a
target outside inference::

    spectrum = build_importance_spectrum(
        catalog,
        source_model=target_source_model,
        merger_rate_fn=target_merger_rate_fn,
        average_mode="analytic_inclination",
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

``gwb_amplitude_marginalized_model`` integrates the amplitude direction
analytically, then draws the amplitude from ``AmplitudeConditional`` under
``handlers.block`` so that draw's density does not enter the NUTS potential.
``reconstruct_amplitude`` consumes the published statistics to recover
independent joint amplitude/shape draws from a saved chain without replaying
the spectrum. NUTS should wrap the model with :func:`hide_amplitude_draws`:
the in-model draw uses a fixed RNG key inside NumPyro's potential/postprocess,
so collecting it would be a single quantile of :math:`p(\varphi\mid\theta,d)`,
not an independent posterior sample.

End-to-end sketch (toy data; runnable as-is):

.. code-block:: python

    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro.distributions as dist
    import xarray as xr
    from numpyro.infer import MCMC, NUTS

    from astrogwb.distributions.amplitude import quadrature_grid
    from astrogwb.detector import gaussian_bin_scale
    from astrogwb.sampling import (
        gwb_amplitude_marginalized_model,
        hide_amplitude_draws,
        reconstruct_amplitude,
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

    # --- Inference: NUTS on the amplitude-marginalized model. Hide the
    # reconstructed amplitude sites so they are not collected from the
    # postprocess replay's fixed-key draw.
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
        merger_rate_amplitude_fn=h0_merger_rate_amplitude,
        priors={"log10_rate": dist.Uniform(-8.0, -6.0)},
    )
    mcmc = MCMC(NUTS(hide_amplitude_draws(model, "H0")), num_warmup=50,
                num_samples=50, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))

    # --- Reconstruction: independent draws from the chain's statistics.
    chain_samples = mcmc.get_samples(group_by_chain=True)
    draws = reconstruct_amplitude(
        jax.random.fold_in(jax.random.PRNGKey(0), 1),
        amplitude_mle=chain_samples["amplitude_mle"],
        template_optimal_snr=chain_samples["template_optimal_snr"],
        template_merger_rate=chain_samples["template_merger_rate"],
        amplitude_parameter="H0",
        amplitude_fn=h0_amplitude,
        merger_rate_amplitude_fn=h0_merger_rate_amplitude,
        prior=amplitude_prior,
        fiducial=h0_fid,
        grid=grid,
    )

    # --- Merge: every returned site is (chain, draw); assign DataArrays.
    import arviz as az

    idata = az.from_numpyro(mcmc)
    for name, values in draws.items():
        idata.posterior[name] = xr.DataArray(
            np.asarray(values), dims=("chain", "draw")
        )
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro import handlers

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
) -> None:
    """Sample parameters and compare a supplied spectrum to Gaussian observations.

    ``spectral_density_fn(params)`` returns a spectrum of shape ``(F,)`` and
    optional deterministic diagnostics. ``scale`` is the prepared per-bin
    standard deviation, normally ``gaussian_bin_scale(psd, time_years, df)``.
    Use ``priors={}`` for likelihood-only evaluation. The observation site is
    ``spectral_density_obs`` with one frequency event dimension. Diagnostic
    names must not collide with priors or that observation site.
    """
    params = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
    prediction, extras = spectral_density_fn(params)
    for name, value in extras.items():
        numpyro.deterministic(name, value)
    numpyro.sample(
        "spectral_density_obs",
        dist.Normal(prediction, scale).to_event(1),
        obs=observed_spectral_density,
    )


def hide_amplitude_draws(
    model: Callable[..., None], amplitude_parameter: str
) -> Callable[..., None]:
    """Hide reconstructed amplitude sites so NUTS does not collect them.

    The in-model draw of ``amplitude_parameter`` is real, but NumPyro's
    potential and postprocess replay it under a fixed RNG key, which yields a
    single quantile of the conditional rather than an independent sample.
    :func:`reconstruct_amplitude` is the independent-draw path from the
    published statistics.
    """
    return handlers.block(
        model,
        hide=[
            amplitude_parameter,
            "total_merger_rate",
            "quadrature_effective_nodes",
        ],
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
    merger_rate_amplitude_fn: MergerRateAmplitudeFn | None = None,
) -> None:
    """Marginalize a multiplicative parameter of any supplied spectrum.

    Sample shape parameters from ``priors`` and evaluate the spectrum once with
    ``amplitude_parameter`` pinned to ``amplitude_fiducial``. The amplitude
    ratio is ``amplitude_fn(phi) / amplitude_fn(amplitude_fiducial)``; the
    spectrum must factor this way throughout the amplitude prior's support.
    Integrate under ``amplitude_prior`` using ``AmplitudeConditional`` and
    its default quadrature grid when ``amplitude_grid`` is omitted.

    Then draw ``amplitude_parameter`` from that same conditional under
    ``handlers.block``, so the draw's log density does not enter the NUTS
    potential -- the ``amplitude_marginalized_log_likelihood`` factor already
    integrated it. Seeded executions (``Predictive``, a seeded trace) record
    the draw as a deterministic, plus ``total_merger_rate`` when the spectrum
    published ``template_merger_rate`` and ``merger_rate_amplitude_fn`` is
    given, and ``quadrature_effective_nodes``. Unseeded ``log_density`` skips
    the draw, so the joint stays a function of the supplied parameter dict.

    Records ``amplitude_mle``, ``template_optimal_snr``, the returned extras,
    and the ``amplitude_marginalized_log_likelihood`` factor. Diagnostics must
    not collide with priors, these three likelihood-owned names, or the
    reconstructed amplitude sites. Importance callers should rename their
    spectrum's ``total_merger_rate`` to ``template_merger_rate`` before
    returning it; see the module example. Wrap NUTS with
    :func:`hide_amplitude_draws` and use :func:`reconstruct_amplitude` for
    independent draws from a saved chain.

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
    normalization = -jnp.sum(jnp.log(scale) + 0.5 * jnp.log(2.0 * jnp.pi))
    log_likelihood_at_mle = normalization - 0.5 * (
        data_norm - amplitude_mle * data_template
    )
    # The marginalization factor *is* the normalizing constant of the
    # conditional that reconstruction later samples.
    numpyro.factor(
        "amplitude_marginalized_log_likelihood",
        log_likelihood_at_mle + conditional.log_normalizer,
    )

    # Draw phi without adding p(phi | theta, d) to the joint: the factor above
    # already integrated that density. handlers.block hides the sample from
    # NUTS / log_density; a seeded execution still draws and the deterministics
    # below record it. Unseeded log_density has no RNG, so skip.
    phi = None
    with handlers.block(hide=[amplitude_parameter]):
        rng_key = numpyro.prng_key()
        if rng_key is not None:
            phi = jnp.asarray(
                numpyro.sample(amplitude_parameter, conditional, rng_key=rng_key)
            )
    if phi is not None:
        numpyro.deterministic(amplitude_parameter, phi)
        template_rate = extras.get("template_merger_rate")
        if merger_rate_amplitude_fn is not None and template_rate is not None:
            numpyro.deterministic(
                "total_merger_rate",
                template_rate
                * merger_rate_amplitude_fn(phi)
                / merger_rate_amplitude_fn(jnp.asarray(amplitude_fiducial)),
            )
        numpyro.deterministic("quadrature_effective_nodes", conditional.effective_nodes)


def reconstruct_amplitude(
    rng_key: jax.Array,
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
) -> dict[str, jax.Array]:
    r"""Independent amplitude draws from published sufficient statistics.

    The statistics' broadcast shape is the batch shape of
    :class:`~astrogwb.distributions.amplitude.AmplitudeConditional`, so a
    ``(chain, draw)`` input batch yields one independent marginalized-parameter
    draw for every chain element. There is no forward physics here -- no
    catalog, no :math:`(F, N)` contraction -- so the cost is ``O(K)`` per draw
    and reconstruction runs against a saved chain alone.

    Returns a dict with ``amplitude_parameter``, ``total_merger_rate``, and
    ``quadrature_effective_nodes``, the same names the in-model draw publishes.

    Parameters
    ----------
    rng_key:
        PRNG key for the inverse-transform draw.
    amplitude_mle, template_optimal_snr, template_merger_rate:
        Sufficient statistics published by
        :func:`gwb_amplitude_marginalized_model`. Their shapes must broadcast to a
        common batch shape; production reconstruction passes ``(chain, draw)``
        arrays.
    amplitude_parameter:
        Name of the marginalized parameter; becomes the dict key of the
        reconstructed draws (e.g. ``"H0"``).
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
    phi = jnp.asarray(conditional.sample(rng_key))
    return {
        amplitude_parameter: phi,
        "total_merger_rate": template_merger_rate
        * merger_rate_amplitude_fn(phi)
        / merger_rate_amplitude_fn(jnp.asarray(fiducial)),
        "quadrature_effective_nodes": conditional.effective_nodes,
    }
