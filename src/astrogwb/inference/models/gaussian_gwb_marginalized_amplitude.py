r"""Amplitude-marginalized inference over a dimensionless amplitude.

:func:`gwb_amplitude_marginalized_model` integrates one multiplicative
amplitude :math:`A` of an arbitrary :class:`~astrogwb.inference.SpectralDensityFn`
under an A-space prior, and publishes amplitude sufficient statistics instead of
sampling it. The spectrum is evaluated at the template (:math:`A = 1`); the
model knows nothing about the physical parameter :math:`\varphi` behind the
amplitude. The caller builds the A-space prior from a NumPyro ``Transform``
:math:`T` with :func:`~astrogwb.distributions.amplitude.amplitude_prior`, pins
:math:`\varphi` at its fiducial :math:`T(\varphi_{\mathrm{fid}}) = 1` when
evaluating the spectrum, and recovers :math:`\varphi` afterwards by drawing
:math:`A` from
:class:`~astrogwb.distributions.amplitude.AmplitudeConditional` with the
published statistics and applying :math:`T^{-1}`.

The template spectrum's total merger rate is registered as
``total_merger_rate_at_unit_amplitude`` so a template quantity never appears
under the name of a physical one.

End-to-end sketch (toy data; runnable as-is):

.. code-block:: python

    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    import xarray as xr
    from numpyro import handlers
    from numpyro.infer import MCMC, NUTS

    from astrogwb.detector import gaussian_bin_scale
    from astrogwb.distributions.amplitude import (
        AmplitudeConditional,
        amplitude_prior,
        quadrature_grid,
    )
    from astrogwb.inference import gwb_amplitude_marginalized_model
    from astrogwb.populations import amplitude_H0_transform

    # --- One-time setup: the A-space prior and the grid it is marginalized on.
    h0_fid = 70.0
    transform = amplitude_H0_transform(h0_fid)  # A = h0_fid / H0
    h0_prior = dist.Uniform(20.0, 140.0)
    prior = amplitude_prior(h0_prior, transform)
    grid = quadrature_grid(prior, num_nodes=2001)

    def toy_spectrum(params):
        template_rate = 10.0 ** params["log10_rate"] * (h0_fid / params["H0"]) ** 3
        prediction = 0.4 * template_rate * (params["H0"] / h0_fid) ** 2 * jnp.ones(3)
        return prediction, {"total_merger_rate": template_rate}

    fiducials = {"H0": h0_fid, "log10_rate": -7.0}
    observed, _ = toy_spectrum(fiducials)

    # --- Inference: NUTS on the amplitude-marginalized model. H0 is a site of
    # `priors`, conditioned at its fiducial and hidden from the trace, so the
    # spectrum is evaluated at the template.
    model = handlers.block(
        handlers.condition(
            partial(
                gwb_amplitude_marginalized_model,
                spectral_density_fn=toy_spectrum,
                observed_spectral_density=observed,
                scale=gaussian_bin_scale(jnp.ones(3), 1.0, 0.25),
                amplitude_prior=prior,
                amplitude_grid=grid,
                priors={"H0": h0_prior, "log10_rate": dist.Uniform(-8.0, -6.0)},
            ),
            data={"H0": h0_fid},
        ),
        hide=["H0"],
    )
    mcmc = MCMC(NUTS(model), num_warmup=50, num_samples=50, num_chains=1,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))

    # --- Reconstruction: one A per (chain, draw), mapped back with T.inv.
    chain_samples = mcmc.get_samples(group_by_chain=True)
    conditional = AmplitudeConditional(
        chain_samples["amplitude_mle"],
        chain_samples["template_optimal_snr"],
        prior=prior,
        grid=grid,
    )
    amplitude = conditional.sample(jax.random.fold_in(jax.random.PRNGKey(0), 1))
    draws = {
        "H0": transform.inv(amplitude),
        "quadrature_effective_nodes": conditional.effective_nodes,
    }

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

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from astrogwb.distributions.amplitude import AmplitudeConditional
from astrogwb.inference.protocol import SpectralDensityFn

__all__ = ["gwb_amplitude_marginalized_model"]


TEMPLATE_RATE_SITE = "total_merger_rate_at_unit_amplitude"
LIKELIHOOD_SITES = (
    "amplitude_mle",
    "template_optimal_snr",
    "amplitude_marginalized_log_likelihood",
)


def gwb_amplitude_marginalized_model(
    *,
    spectral_density_fn: SpectralDensityFn,
    observed_spectral_density: jax.Array,
    priors: Mapping[str, dist.Distribution],
    scale: jax.Array,
    amplitude_prior: dist.Distribution,
    amplitude_grid: jax.Array | None = None,
    frequency_mask: jax.Array | None = None,
) -> None:
    """Marginalize a multiplicative amplitude of any supplied spectrum.

    Sample every site in ``priors`` and evaluate the spectrum once; that
    spectrum is the template at :math:`A = 1`, so it must already be evaluated
    at the fiducial of whichever physical parameter the amplitude stands for.
    A caller pins that parameter with ``numpyro.handlers.condition`` plus
    ``block`` on a ``priors`` site. Integrate :math:`A` under
    ``amplitude_prior`` -- the A-space pushforward from
    :func:`~astrogwb.distributions.amplitude.amplitude_prior` -- using
    ``AmplitudeConditional`` and its default quadrature grid when
    ``amplitude_grid`` is omitted.

    Records ``amplitude_mle``, ``template_optimal_snr``, the returned extras,
    and the ``amplitude_marginalized_log_likelihood`` factor. The spectrum's
    ``total_merger_rate`` extra, if any, is the template's and is registered as
    ``total_merger_rate_at_unit_amplitude``; other extras are recorded
    unchanged. Diagnostics must not collide with priors or the likelihood-owned
    names. Draw amplitudes afterwards from ``AmplitudeConditional`` using the
    same prior and grid, and map them to the physical parameter with the
    inverse transform.

    ``frequency_mask`` is an optional boolean array of shape ``(F,)`` selecting
    the bins the likelihood counts, as in
    :func:`~astrogwb.inference.models.gaussian_gwb_model.gwb_spectral_density_model`.
    Every sum below restricts to it.

    Raises ``ValueError`` if a diagnostic name collides with a sampled site, a
    likelihood-owned site, or another published diagnostic. All spectrum, observation, and scale arrays have
    shape ``(F,)``.
    """
    params = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
    model_spectral_density, extras = spectral_density_fn(params)
    reserved = {*priors, *LIKELIHOOD_SITES}
    published: dict[str, ArrayLike] = {}
    for name, value in extras.items():
        site = TEMPLATE_RATE_SITE if name == "total_merger_rate" else name
        if site in reserved or site in published:
            raise ValueError(f"spectrum diagnostic {site!r} collides with a model site")
        published[site] = value
    for name, value in published.items():
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
        prior=amplitude_prior,
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
