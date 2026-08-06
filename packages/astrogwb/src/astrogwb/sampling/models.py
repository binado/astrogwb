from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp

from astrogwb.detector import gaussian_bin_scale
from astrogwb.gwb import AverageMode, spectral_density
from astrogwb.importance.diagnostics import relative_ess
from astrogwb.importance.protocol import MergerRateAndLogWeightsFn
from astrogwb.sampling.amplitude import (
    AmplitudeQuadrature,
    amplitude_log_integrand,
    amplitude_statistics,
    best_fit_residual,
    gaussian_log_norm,
)


def _predicted_spectral_density(
    *,
    frequencies: jax.Array,
    polarization_power: jax.Array,
    samples: Mapping[str, jax.Array],
    observed_spectral_density: jax.Array,
    effective_psd: jax.Array,
    observation_time: float,
    average_mode: AverageMode,
    merger_rate_and_log_weights_fn: MergerRateAndLogWeightsFn,
    priors: Mapping[str, dist.Distribution],
    constants: Mapping[str, Any],
    frequency_mask: jax.Array | None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, float | jax.Array, jax.Array]:
    """Sample the priors and contract the catalog into a predicted spectrum.

    The body shared by every model in this module: it registers one
    ``numpyro.sample`` site per prior, invokes the catalog callback, and applies
    ``frequency_mask``. Callers register whichever deterministics and likelihood
    they need on top. ``overrides`` is merged last into ``params``, which is how
    the amplitude-marginalized model pins its amplitude parameter to the
    reference value.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, float | jax.Array, jax.Array]
        ``(model_spectral_density, observed_spectral_density, scale,
        total_merger_rate, log_weights)``. The first three are masked; the last
        two are as returned by the callback.
    """
    sampled_params = {
        name: numpyro.sample(name, prior) for name, prior in priors.items()
    }
    params = {**constants, **sampled_params, **(overrides or {})}

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

    scale = gaussian_bin_scale(effective_psd, frequencies, observation_time)
    if frequency_mask is not None:
        model_spectral_density = model_spectral_density[frequency_mask]
        observed_spectral_density = observed_spectral_density[frequency_mask]
        scale = scale[frequency_mask]

    return (
        model_spectral_density,
        observed_spectral_density,
        scale,
        total_merger_rate,
        log_weights,
    )


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
    frequency_mask: jax.Array | None = None,
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
        Frequency grid in Hz, shape ``(F,)``.
    polarization_power:
        Per-source polarization power at each frequency, shape ``(F, N)`` where
        ``N`` is the catalog size.
    samples:
        Catalog arrays passed to ``merger_rate_and_log_weights_fn``. Each value
        should have leading dimension ``N``.
    observed_spectral_density:
        Observed SGWB spectral density at ``frequencies``, shape ``(F,)``.
        Must be the full frequency grid even when ``frequency_mask`` is set;
        masking is applied inside the model.
    effective_psd:
        Network effective power spectral density at ``frequencies``, shape
        ``(F,)``. Must match ``frequencies``; masked internally when
        ``frequency_mask`` is set.
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
    frequency_mask:
        Optional boolean mask of shape ``(F,)``. When provided, only masked
        bins from the full-length arrays above enter the likelihood.
    """
    (
        model_spectral_density,
        observed_spectral_density,
        scale,
        total_merger_rate,
        log_weights,
    ) = _predicted_spectral_density(
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=observed_spectral_density,
        effective_psd=effective_psd,
        observation_time=observation_time,
        average_mode=average_mode,
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        priors=priors or {},
        constants=constants or {},
        frequency_mask=frequency_mask,
    )

    numpyro.deterministic("total_merger_rate", total_merger_rate)
    numpyro.deterministic("importance_relative_ess", relative_ess(log_weights))

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
    quadrature: AmplitudeQuadrature,
    priors: Mapping[str, dist.Distribution] | None = None,
    constants: Mapping[str, Any] | None = None,
    frequency_mask: jax.Array | None = None,
) -> None:
    r"""SGWB model with a multiplicative amplitude marginalized out of the likelihood.

    Identical to :func:`spectral_density_model` except that one strictly
    multiplicative parameter is integrated out instead of being sampled, which
    removes the long, curved amplitude--shape degeneracy that NUTS handles
    worst. The marginalization is essentially free here: the amplitude never
    touches the importance weights, and ``polarization_power`` is a fixed
    precomputed catalog.

    The physical parameter :math:`\varphi` is marginalized numerically on the
    fixed grid in ``quadrature`` (built by
    :func:`~astrogwb.sampling.amplitude.make_amplitude_quadrature`) under its
    own prior :math:`\pi(\varphi)`, for an arbitrary scaling
    :math:`A = f(\varphi)` to the multiplicative amplitude. What
    :func:`~astrogwb.sampling.amplitude.draw_marginalized_parameter` returns in
    post-processing is :math:`\varphi` itself (e.g. :math:`H_0`), not
    :math:`A`. The only error is quadrature error, so grid resolution should be
    checked with :func:`~astrogwb.sampling.amplitude.quadrature_effective_nodes`.

    The callback is invoked with ``amplitude_parameter`` pinned to
    ``fiducials[amplitude_parameter]``, so the predicted spectrum it returns is
    the *template* :math:`\mathbf{m}(\theta)` and the marginalized amplitude
    :math:`A` is the dimensionless ratio to that reference. For a parameter the
    spectrum is linear in -- ``local_merger_rate``, say -- the physical value is
    recovered as :math:`A \times` ``fiducials[amplitude_parameter]``, and the
    existing reference callback works unchanged. A parameter that enters
    inversely, such as :math:`H_0` with :math:`S_h \propto H_0^{-1}`, needs a
    callback that declares its own synthetic amplitude key, because the inverse
    map cannot reuse a physical parameter name.

    Registered sites:

    - one ``numpyro.sample`` per entry in ``priors``;
    - ``amplitude_ml``, ``template_optimal_snr``, and
      ``importance_relative_ess`` as deterministics;
    - ``amplitude_marginalized_log_likelihood`` as a ``numpyro.factor``.

    ``total_merger_rate`` is deliberately *not* registered: at a pinned
    amplitude it would be the rate at unit amplitude, a different quantity under
    the same name. Use :func:`spectral_density_model` when it is needed.

    The two amplitude statistics are what post-processing needs to reconstruct
    joint :math:`(\varphi, \theta)` samples via
    :func:`astrogwb.sampling.amplitude.draw_marginalized_parameter` --
    ``factor`` sites do not appear in ArviZ's posterior group, so they must be
    carried explicitly.

    Parameters
    ----------
    amplitude_parameter:
        Name of the parameter to marginalize over. Must be understood by
        ``merger_rate_and_log_weights_fn`` and must not appear in ``priors``.
    fiducials:
        Fiducial hyperparameters; ``fiducials[amplitude_parameter]`` is the
        reference value that defines the template.
    quadrature:
        Precomputed grid from
        :func:`~astrogwb.sampling.amplitude.make_amplitude_quadrature` that
        marginalizes the amplitude direction out of the Gaussian likelihood.

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

    (
        model_spectral_density,
        observed_spectral_density,
        scale,
        _,
        log_weights,
    ) = _predicted_spectral_density(
        frequencies=frequencies,
        polarization_power=polarization_power,
        samples=samples,
        observed_spectral_density=observed_spectral_density,
        effective_psd=effective_psd,
        observation_time=observation_time,
        average_mode=average_mode,
        merger_rate_and_log_weights_fn=merger_rate_and_log_weights_fn,
        priors=priors,
        constants=constants or {},
        frequency_mask=frequency_mask,
        overrides={amplitude_parameter: fiducials[amplitude_parameter]},
    )

    amplitude_ml, template_optimal_snr = amplitude_statistics(
        model_spectral_density,
        observed_spectral_density,
        scale,
    )
    numpyro.deterministic("amplitude_ml", amplitude_ml)
    numpyro.deterministic("template_optimal_snr", template_optimal_snr)
    numpyro.deterministic("importance_relative_ess", relative_ess(log_weights))

    log_integrand = amplitude_log_integrand(
        amplitude_ml, template_optimal_snr, quadrature=quadrature
    )
    numpyro.factor(
        "amplitude_marginalized_log_likelihood",
        gaussian_log_norm(scale)
        - best_fit_residual(
            model_spectral_density,
            observed_spectral_density,
            scale,
            amplitude_ml=amplitude_ml,
        )
        + logsumexp(log_integrand, axis=-1)
        - quadrature.log_prior_mass,
    )
