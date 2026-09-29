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
        density_sites=DEFAULT_DENSITY_SITES,
    )[0]
    model = partial(
        gwb_spectral_density_model,
        spectral_density_fn=spectrum,
        observed_spectral_density=observed,
        priors=priors,
        scale=gaussian_bin_scale(effective_psd, observation_time, df),
    )

:func:`~astrogwb.inference.models.gaussian_gwb_marginalized_amplitude.gwb_amplitude_marginalized_model`
consumes the same spectrum callable; see that module for the marginalized
variant and its end-to-end sketch.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import numpyro
import numpyro.distributions as dist

from astrogwb.inference.protocol import SpectralDensityFn

__all__ = ["gwb_spectral_density_model"]


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
