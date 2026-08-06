r"""Numerical marginalization of a multiplicative amplitude direction.

Under the per-frequency Gaussian likelihood used by
:func:`~astrogwb.sampling.models.spectral_density_model`, one parameter can
enter the predicted spectrum as a pure multiplicative factor,

.. math:: \boldsymbol{\mu}(\varphi, \theta) = f(\varphi)\, \mathbf{m}(\theta)

with :math:`\mathbf{m}(\theta)` the *template* -- the spectrum evaluated at a
fixed reference value of the marginalized parameter -- and :math:`f` an
arbitrary scaling from the physical parameter :math:`\varphi` (e.g. a merger
rate, or :math:`H_0` through :math:`f(H_0) = H_{0,\mathrm{fid}}/H_0`) to the
dimensionless multiplicative amplitude :math:`A = f(\varphi)`. Define the
noise-weighted inner product :math:`(x|y) = \sum_i x_i y_i / \sigma_i^2`. Then

.. math::

    \hat{A} = \frac{(d|m)}{(m|m)}, \qquad \rho = \sqrt{(m|m)},

are the maximum-likelihood amplitude and the template optimal SNR, and
completing the square in :math:`A` gives

.. math::

    -\tfrac{1}{2}\sum_i \left(\frac{d_i - A\, m_i}{\sigma_i}\right)^2
    = -R - \tfrac{1}{2}\rho^2 (A - \hat{A})^2,

with :math:`R = \tfrac{1}{2}\sum_i((d_i - \hat{A}m_i)/\sigma_i)^2` the
best-fit residual. This module marginalizes :math:`\varphi` numerically on a
fixed 1D grid under the caller's actual prior :math:`\pi(\varphi)`, rather
than requiring the prior to be stated on :math:`A` itself. Folding the
trapezoid weights into a precomputed log-measure turns the integral over the
grid into a single ``logsumexp``. With :math:`w_k` the trapezoid weights at
grid node :math:`\varphi_k`,

.. math::

    \ln Z = \ln \mathcal{N}_d - R
        + \operatorname{logsumexp}_k\!\left[\ell_k - \tfrac{1}{2}\bigl(\rho(f_k - \hat{A})\bigr)^2\right]
        - \ln \Pi,

.. math::

    \ell_k = \ln\pi(\varphi_k) + \ln w_k, \qquad
    \ln \Pi = \operatorname{logsumexp}_k\, \ell_k,

where :math:`f_k = f(\varphi_k)`, :math:`\ln \mathcal{N}_d` is the Gaussian
normalization, and :math:`\ln \Pi` normalizes the prior mass on the grid, so
``log_prior`` may be passed unnormalized. Squaring :math:`\rho(f_k - \hat{A})`
rather than forming :math:`\rho^2(f_k-\hat A)^2` avoids overflowing
:math:`\rho^2` at very high SNR, and :math:`R` is computed from residuals
directly rather than as :math:`\tfrac{1}{2}(d|d) - \tfrac{1}{2}\hat{A}^2\rho^2`
-- those two terms are each :math:`\sim \mathrm{SNR}^2/2` and nearly cancel at
high SNR.

"Exact up to quadrature error" only holds if the grid resolves the conditional
posterior, whose width in :math:`\varphi` is :math:`\sigma_A/|f'(\varphi)|`.
No quadrature rule rescues a Gaussian bump spanning three nodes, so grid
adequacy must be checked with :func:`quadrature_effective_nodes`, not assumed.

All functions broadcast over leading batch dimensions and contract over the
trailing frequency axis, so post-processing can feed them ``(chain, draw)``
shaped arrays directly.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from astrogwb.importance.diagnostics import relative_ess

_LOG_TWO_PI = math.log(2.0 * math.pi)


class AmplitudeScalingFn(Protocol):
    """Map the marginalized parameter to the multiplicative amplitude :math:`A = f(\\varphi)`."""

    def __call__(self, marginalized_parameter: jax.Array) -> jax.Array: ...


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

    The constant that makes the log evidence assembled in
    :func:`~astrogwb.sampling.models.amplitude_marginalized_model` a genuine
    log marginal likelihood rather than a log density up to an additive
    constant.
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


class AmplitudeQuadrature(NamedTuple):
    """Precomputed grid, scaling, and prior measure for numerical marginalization.

    Built once by :func:`make_amplitude_quadrature` and then reused every MCMC
    step; a plain ``NamedTuple`` keeps it a JAX pytree without needing to be
    stored as model state, since the callable that built it is only consumed
    at construction time.
    """

    grid: jax.Array
    """``(K,)`` values of the marginalized parameter :math:`\\varphi`."""

    amplitude: jax.Array
    """``(K,)`` :math:`f(\\varphi_k)`, the multiplicative factor at each node."""

    log_measure: jax.Array
    """``(K,)`` :math:`\\ln\\pi(\\varphi_k) + \\ln w_k`, trapezoid-weighted log prior."""

    log_prior_mass: jax.Array
    """Scalar ``logsumexp(log_measure)``, the normalization for the prior mass
    on the grid; :func:`~astrogwb.sampling.models.amplitude_marginalized_model`
    subtracts it from the assembled log evidence."""


def make_amplitude_quadrature(
    *,
    grid: jax.Array,
    log_prior: jax.Array,
    scaling: AmplitudeScalingFn,
) -> AmplitudeQuadrature:
    """Build the fixed quadrature grid consumed by every MCMC step.

    Called once, before sampling starts, with concrete (non-traced) arrays --
    like :func:`astrogwb.importance.models.bns_madau_dickinson_modified_propagation.make_merger_rate_and_log_weights_fn`,
    this is a factory, not something invoked inside ``jax.jit``.

    Parameters
    ----------
    grid:
        Strictly increasing 1D array of :math:`\\varphi` values, length at
        least 2.
    log_prior:
        :math:`\\ln\\pi(\\varphi_k)` at each grid node, same shape as ``grid``.
        May be unnormalized; ``log_prior_mass`` reports the resulting offset
        and the model normalizes it away.
    scaling:
        Maps the grid to the multiplicative amplitude, :math:`f(\\varphi_k)`.

    Raises
    ------
    ValueError
        If ``grid`` is not 1D, has fewer than 2 points, is not strictly
        increasing, or does not match the shape of ``log_prior``.
    """
    grid = jnp.asarray(grid)
    log_prior = jnp.asarray(log_prior)
    if grid.ndim != 1:
        raise ValueError(f"grid must be 1D, got shape {grid.shape}")
    if grid.shape[0] < 2:
        raise ValueError(f"grid must have at least 2 points, got {grid.shape[0]}")
    if log_prior.shape != grid.shape:
        raise ValueError(
            f"log_prior shape {log_prior.shape} must match grid shape {grid.shape}"
        )
    dx = jnp.diff(grid)
    if not bool(jnp.all(dx > 0)):
        raise ValueError("grid must be strictly increasing")

    weights = 0.5 * jnp.concatenate([dx[:1], dx[1:] + dx[:-1], dx[-1:]])
    log_measure = log_prior + jnp.log(weights)
    return AmplitudeQuadrature(
        grid=grid,
        amplitude=scaling(grid),
        log_measure=log_measure,
        log_prior_mass=logsumexp(log_measure),
    )


def amplitude_log_integrand(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    quadrature: AmplitudeQuadrature,
) -> jax.Array:
    """Shared ``(..., K)`` log-integrand behind the model factor and the two
    functions below.

    Routing :func:`~astrogwb.sampling.models.amplitude_marginalized_model`,
    :func:`draw_marginalized_parameter`, and :func:`quadrature_effective_nodes`
    through one implementation is what keeps them from drifting apart -- in
    particular, it guarantees the inverse-transform sampler in
    :func:`draw_marginalized_parameter` draws from exactly the density the
    model factor integrated.
    """
    scaled_residual = jnp.expand_dims(template_optimal_snr, -1) * (
        quadrature.amplitude - jnp.expand_dims(amplitude_ml, -1)
    )
    return quadrature.log_measure - 0.5 * scaled_residual**2


def draw_marginalized_parameter(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    quadrature: AmplitudeQuadrature,
    rng_key: jax.Array,
) -> jax.Array:
    r"""Draw one value of :math:`\varphi` per posterior sample of :math:`\theta`.

    Inverse-transform sampling on the grid: the CDF is the cumulative sum of
    the same ``log_measure``-weighted nodes that
    :func:`amplitude_log_integrand` integrates, *not* a separately-computed
    cumulative trapezoid, so it terminates at exactly the :math:`Z` that was
    marginalized and the draws follow precisely that density rather than an
    :math:`O(\Delta^2)`-nearby one.

    Leading ``(chain, draw)`` dimensions of the statistics are preserved and
    the function is fully broadcast (no ``vmap``), so it is shape-preserving
    and un-chunked by default. That materializes an ``(..., K)`` array: at
    ``K=4000`` grid nodes and 20k posterior samples that is already ~640 MB in
    float64. Because this runs in post-processing, outside ``jax.jit``, chunk
    it in plain Python if memory is tight -- loop over the chain axis, or
    reshape the leading dimensions and slice, calling this function once per
    chunk and concatenating the results.

    Parameters
    ----------
    amplitude_ml, template_optimal_snr:
        Sufficient statistics from :func:`amplitude_statistics`.
    quadrature:
        Precomputed grid from :func:`make_amplitude_quadrature`.
    rng_key:
        PRNG key; one uniform draw is consumed per leading-dimension element.

    Returns
    -------
    jax.Array
        :math:`\varphi` draws, same leading shape as ``amplitude_ml``, clipped
        to ``[grid[0], grid[-1]]``.
    """
    log_integrand = amplitude_log_integrand(
        amplitude_ml, template_optimal_snr, quadrature=quadrature
    )
    shifted = jnp.exp(log_integrand - jnp.max(log_integrand, axis=-1, keepdims=True))
    cdf = jnp.cumsum(shifted, axis=-1)
    cdf = cdf / cdf[..., -1:]

    grid = quadrature.grid
    num_nodes = grid.shape[0]
    u = jax.random.uniform(rng_key, shape=amplitude_ml.shape)
    idx = jnp.clip(jnp.sum(cdf < u[..., None], axis=-1), 1, num_nodes - 1)

    cdf_hi = jnp.take_along_axis(cdf, idx[..., None], axis=-1)[..., 0]
    cdf_lo = jnp.take_along_axis(cdf, (idx - 1)[..., None], axis=-1)[..., 0]
    grid_hi = grid[idx]
    grid_lo = grid[idx - 1]

    # Deep in the tails `shifted` underflows to 0, so the CDF has long flat
    # plateaus; guard the division so those draws land at `grid_lo` instead of
    # NaN from 0/0.
    span = jnp.where(cdf_hi > cdf_lo, cdf_hi - cdf_lo, 1.0)
    fraction = jnp.where(cdf_hi > cdf_lo, (u - cdf_lo) / span, 0.0)
    return grid_lo + fraction * (grid_hi - grid_lo)


def quadrature_effective_nodes(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    quadrature: AmplitudeQuadrature,
) -> jax.Array:
    r"""Grid-adequacy diagnostic: how many nodes actually carry the conditional posterior.

    Reuses :func:`~astrogwb.importance.diagnostics.relative_ess` on the
    log-integrand -- the same Kish effective-sample-size construction used for
    importance weights -- and rescales it by :math:`K` so the result is a node
    count rather than a fraction. A Gaussian conditional posterior spanning
    only a handful of grid nodes will report a small value here even though
    the assembled log evidence looks finite and plausible; this should be
    comfortably above approximately 30.
    """
    log_integrand = amplitude_log_integrand(
        amplitude_ml, template_optimal_snr, quadrature=quadrature
    )
    return relative_ess(log_integrand) * quadrature.grid.shape[0]
