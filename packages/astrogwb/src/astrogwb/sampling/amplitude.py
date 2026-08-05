r"""Analytic marginalization of a strictly multiplicative amplitude.

Under the per-frequency Gaussian likelihood used by
:func:`~astrogwb.sampling.models.spectral_density_model`, a parameter that
enters the predicted spectrum as a pure multiplicative factor can be integrated
out exactly. Write the prediction as

.. math:: \boldsymbol{\mu}(A, \theta) = A\, \mathbf{m}(\theta)

with :math:`\mathbf{m}(\theta)` the *template* -- the spectrum evaluated at a
fixed reference value of the amplitude parameter -- so that :math:`A` is the
dimensionless ratio to that reference. Define the noise-weighted inner product
:math:`(x|y) = \sum_i x_i y_i / \sigma_i^2`. Then

.. math::

    \hat{A} = \frac{(d|m)}{(m|m)}, \qquad \rho = \sqrt{(m|m)},

are the maximum-likelihood amplitude and the template optimal SNR, and the
conditional amplitude uncertainty at fixed :math:`\theta` is
:math:`\sigma_A = 1/\rho`.

For a prior with precision :math:`\tau_0` (``1/scale**2`` for a Normal, ``0``
for a Uniform), location :math:`\mu_0`, and support :math:`[\ell, h]`
(:math:`\pm\infty` for a Normal), completing the square gives

.. math::

    \tau = \rho^2 + \tau_0, \qquad
    \tilde{m} = \frac{\rho^2 \hat{A} + \tau_0 \mu_0}{\tau}, \qquad
    Q = \rho^2 \hat{A}^2 + \tau_0 \mu_0^2 - \tau \tilde{m}^2,

so that the conditional posterior is :math:`\mathcal{N}(\tilde{m}, \tau^{-1})`
truncated to the prior support, and the log marginal likelihood is

.. math::

    \ln Z = \ln \mathcal{N}_d - R - \tfrac{1}{2} Q + \ln \mathcal{N}_\pi
            + \tfrac{1}{2}\ln\frac{2\pi}{\tau}
            + \ln\left[\Phi(\beta) - \Phi(\alpha)\right],

with :math:`R = \tfrac{1}{2}\sum_i ((d_i - \hat{A} m_i)/\sigma_i)^2` the
best-fit residual, :math:`\ln \mathcal{N}_d` and :math:`\ln \mathcal{N}_\pi` the
Gaussian and prior normalizations, and
:math:`\alpha, \beta = (\ell - \tilde{m})\sqrt{\tau}, (h - \tilde{m})\sqrt{\tau}`.

Two deliberate choices about how this is evaluated:

- :math:`R` is computed as the residual sum of squares, *not* as
  :math:`\tfrac{1}{2}(d|d) - \tfrac{1}{2}\hat{A}^2\rho^2`. Those two terms are
  each :math:`\sim \mathrm{SNR}^2/2` and nearly cancel at high SNR.
- the truncation term is evaluated in log space in whichever Gaussian tail is
  smaller. The naive ``log(Phi(beta) - Phi(alpha))`` underflows to ``-inf`` when
  warmup wanders somewhere :math:`\hat{A}` sits many :math:`\sigma` outside the
  prior support, which poisons the whole chain rather than just rejecting the
  step.

All functions broadcast over leading batch dimensions and contract over the
trailing frequency axis, so post-processing can feed them ``(chain, draw)``
shaped arrays directly.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr

# Priors on the amplitude that admit a closed-form marginalization.
type AmplitudePrior = dist.Normal | dist.Uniform

_LOG_TWO_PI = math.log(2.0 * math.pi)


def noise_weighted_inner_product(
    x: jax.Array,
    y: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    r"""Noise-weighted inner product :math:`(x|y) = \sum_i x_i y_i / \sigma_i^2`.

    Parameters
    ----------
    x, y:
        Arrays whose trailing axis runs over frequency bins.
    scale:
        Per-bin Gaussian noise scale :math:`\sigma_i`, same trailing axis.

    Returns
    -------
    jax.Array
        The contraction over the trailing axis; leading batch dimensions
        broadcast.
    """
    return jnp.sum(x * y / scale**2, axis=-1)


def amplitude_statistics(
    model_spectral_density: jax.Array,
    observed_spectral_density: jax.Array,
    scale: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Maximum-likelihood amplitude and template optimal SNR.

    These are the sufficient statistics of the amplitude-marginalized
    likelihood: everything downstream depends on the data and the template only
    through :math:`\hat{A} = (d|m)/(m|m)` and :math:`\rho = \sqrt{(m|m)}`. They
    are preferred over the raw inner products because they are better
    conditioned, directly interpretable (:math:`\sigma_A = 1/\rho`), and
    invertible by multiplication alone: :math:`(m|m) = \rho^2` and
    :math:`(d|m) = \hat{A}\rho^2`.

    Parameters
    ----------
    model_spectral_density:
        Template :math:`\mathbf{m}(\theta)`, i.e. the predicted spectrum at the
        reference amplitude.
    observed_spectral_density:
        Observed spectrum :math:`\mathbf{d}`.
    scale:
        Per-bin Gaussian noise scale.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(amplitude_ml, template_optimal_snr)``.
    """
    template_norm = noise_weighted_inner_product(
        model_spectral_density, model_spectral_density, scale
    )
    data_template = noise_weighted_inner_product(
        observed_spectral_density, model_spectral_density, scale
    )
    return data_template / template_norm, jnp.sqrt(template_norm)


def gaussian_log_norm(scale: jax.Array) -> jax.Array:
    r"""Normalization :math:`-\sum_i \ln \sigma_i - \frac{n}{2}\ln 2\pi`.

    The constant that makes :func:`amplitude_log_evidence` a genuine log
    marginal likelihood rather than a log density up to an additive constant.
    """
    return -jnp.sum(jnp.log(scale), axis=-1) - 0.5 * scale.shape[-1] * _LOG_TWO_PI


def best_fit_residual(
    model_spectral_density: jax.Array,
    observed_spectral_density: jax.Array,
    scale: jax.Array,
    *,
    amplitude_ml: jax.Array,
) -> jax.Array:
    r"""Half the chi-square at the best-fit amplitude.

    :math:`R = \frac{1}{2}\sum_i ((d_i - \hat{A} m_i)/\sigma_i)^2`, computed
    directly from the residuals to avoid the catastrophic cancellation of the
    algebraically equivalent :math:`\frac{1}{2}(d|d) - \frac{1}{2}\hat{A}^2
    \rho^2` at high SNR.
    """
    residual = (
        observed_spectral_density
        - jnp.expand_dims(amplitude_ml, -1) * model_spectral_density
    ) / scale
    return 0.5 * jnp.sum(residual**2, axis=-1)


class _ConditionalTerms(NamedTuple):
    """Completed-square terms shared by the conditional and the evidence."""

    precision: jax.Array
    mean: jax.Array
    quadratic_leftover: jax.Array
    low: jax.Array
    high: jax.Array
    log_prior_norm: jax.Array
    truncated: bool


def _conditional_terms(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    prior: AmplitudePrior,
) -> _ConditionalTerms:
    """Derive ``(tau, m_tilde, Q)`` and the prior support in one place.

    Both :func:`amplitude_conditional` and :func:`amplitude_log_evidence` go
    through here, so a sign fixed in one is fixed in the other.
    """
    if isinstance(prior, dist.Normal):
        prior_scale = jnp.asarray(prior.scale)
        prior_loc = jnp.asarray(prior.loc)
        prior_precision = 1.0 / prior_scale**2
        low = jnp.asarray(-jnp.inf)
        high = jnp.asarray(jnp.inf)
        log_prior_norm = -jnp.log(prior_scale) - 0.5 * _LOG_TWO_PI
        truncated = False
    elif isinstance(prior, dist.Uniform):
        low = jnp.asarray(prior.low)
        high = jnp.asarray(prior.high)
        prior_precision = jnp.asarray(0.0)
        prior_loc = jnp.asarray(0.0)
        log_prior_norm = -jnp.log(high - low)
        truncated = True
    else:
        raise TypeError(
            "amplitude prior must be a numpyro Normal or Uniform distribution, "
            f"got {type(prior).__name__}"
        )

    data_precision = template_optimal_snr**2
    precision = data_precision + prior_precision
    mean = (data_precision * amplitude_ml + prior_precision * prior_loc) / precision
    quadratic_leftover = (
        data_precision * amplitude_ml**2
        + prior_precision * prior_loc**2
        - precision * mean**2
    )
    return _ConditionalTerms(
        precision=precision,
        mean=mean,
        quadratic_leftover=quadratic_leftover,
        low=low,
        high=high,
        log_prior_norm=log_prior_norm,
        truncated=truncated,
    )


def _log_sub_exp(larger: jax.Array, smaller: jax.Array) -> jax.Array:
    """``log(exp(larger) - exp(smaller))`` for ``smaller <= larger``.

    The guard keeps the result NaN-free (and hence differentiable) when the two
    arguments coincide, which would otherwise produce ``log`` of a negative
    rounding residual.
    """
    delta = smaller - larger
    safe_delta = jnp.where(delta < 0.0, delta, -jnp.inf)
    return larger + jnp.log1p(-jnp.exp(safe_delta))


def _log_gauss_mass(a: jax.Array, b: jax.Array) -> jax.Array:
    r"""``log(Phi(b) - Phi(a))`` evaluated in the smaller tail.

    Both forms below are mathematically exact for any ``a <= b``; the choice
    only decides which one avoids subtracting two numbers close to 1.
    """
    lower_tail = _log_sub_exp(log_ndtr(b), log_ndtr(a))
    upper_tail = _log_sub_exp(log_ndtr(-a), log_ndtr(-b))
    return jnp.where(b <= 0.0, lower_tail, upper_tail)


def amplitude_conditional(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    prior: AmplitudePrior,
) -> dist.Distribution:
    r"""Conditional posterior :math:`p(A \mid \theta, \mathbf{d})`.

    A Normal :math:`\mathcal{N}(\tilde{m}, \tau^{-1})`, truncated to the prior
    support when ``prior`` is Uniform. Its batch shape follows the leading
    dimensions of ``amplitude_ml``.

    Parameters
    ----------
    amplitude_ml:
        Maximum-likelihood amplitude :math:`\hat{A}` from
        :func:`amplitude_statistics`.
    template_optimal_snr:
        Template optimal SNR :math:`\rho` from :func:`amplitude_statistics`.
    prior:
        The amplitude prior the chain was run under.
    """
    terms = _conditional_terms(amplitude_ml, template_optimal_snr, prior)
    scale = 1.0 / jnp.sqrt(terms.precision)
    if terms.truncated:
        return dist.TruncatedNormal(terms.mean, scale, low=terms.low, high=terms.high)
    return dist.Normal(terms.mean, scale)


def amplitude_log_evidence(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    prior: AmplitudePrior,
    residual: jax.Array,
    log_norm: jax.Array,
) -> jax.Array:
    r"""Log marginal likelihood :math:`\ln \int p(\mathbf{d} \mid A, \theta) \pi(A)\,dA`.

    Includes the full Gaussian and prior normalizations, so the result is a real
    log marginal likelihood and is directly comparable against the potential
    energy of :func:`~astrogwb.sampling.models.spectral_density_model`.

    Parameters
    ----------
    amplitude_ml, template_optimal_snr:
        Sufficient statistics from :func:`amplitude_statistics`.
    prior:
        Amplitude prior. Normal marginalizes over the whole real line; Uniform
        contributes the truncation term.
    residual:
        Best-fit residual :math:`R` from :func:`best_fit_residual`.
    log_norm:
        Gaussian normalization from :func:`gaussian_log_norm`.
    """
    terms = _conditional_terms(amplitude_ml, template_optimal_snr, prior)
    log_evidence = (
        log_norm
        + terms.log_prior_norm
        - residual
        - 0.5 * terms.quadratic_leftover
        + 0.5 * (_LOG_TWO_PI - jnp.log(terms.precision))
    )
    if terms.truncated:
        sqrt_precision = jnp.sqrt(terms.precision)
        alpha = (terms.low - terms.mean) * sqrt_precision
        beta = (terms.high - terms.mean) * sqrt_precision
        log_evidence = log_evidence + _log_gauss_mass(alpha, beta)
    return log_evidence


def draw_amplitude(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    prior: AmplitudePrior,
    rng_key: jax.Array,
) -> jax.Array:
    r"""Draw one amplitude per posterior sample of :math:`\theta`.

    Pairing each draw with its :math:`\theta` reconstructs samples from the full
    joint posterior :math:`p(A, \theta \mid \mathbf{d})` that the marginalized
    chain never explored directly. Leading ``(chain, draw)`` dimensions of the
    statistics are preserved, so an ``InferenceData`` posterior group can be fed
    in as-is.

    Note that the amplitude prior used at run time acts as a *proposal*: if the
    physical parameter is a nonlinear function of :math:`A`, the scientific
    prior must be applied afterwards by reweighting (see
    :func:`astrogwb.importance.diagnostics.log_prior_reweighting`).
    """
    conditional = amplitude_conditional(amplitude_ml, template_optimal_snr, prior=prior)
    return jnp.asarray(conditional.sample(rng_key))
