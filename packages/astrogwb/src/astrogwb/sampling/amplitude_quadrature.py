r"""Numerical marginalization of an amplitude that enters non-multiplicatively.

:mod:`astrogwb.sampling.amplitude` marginalizes a strictly multiplicative
amplitude :math:`A` in closed form, which forces the prior to be stated on
:math:`A` itself. When the physical parameter :math:`\varphi` enters through an
arbitrary scaling :math:`A = f(\varphi)` -- :math:`H_0`, say, with
:math:`f(\varphi) = H_{0,\mathrm{fid}}/\varphi` -- a prior on :math:`A` is not
the prior on :math:`\varphi`, and reweighting after the fact
(:func:`astrogwb.importance.diagnostics.log_prior_reweighting`) is only exact
in the infinite-sample limit.

This module marginalizes :math:`\varphi` numerically instead, on a fixed 1D
grid under the caller's actual prior :math:`\pi(\varphi)`. Writing out the
per-frequency Gaussian likelihood and completing the square in :math:`A`
exactly as :mod:`astrogwb.sampling.amplitude` does gives

.. math::

    -\tfrac{1}{2}\sum_i \left(\frac{d_i - A\, g_i}{\sigma_i}\right)^2
    = -R - \tfrac{1}{2}\rho^2 (A - \hat{A})^2,

with :math:`\hat{A} = (d|g)/(g|g)`, :math:`\rho = \sqrt{(g|g)}`, and
:math:`R = \tfrac{1}{2}\sum_i((d_i - \hat{A}g_i)/\sigma_i)^2` -- precisely
:func:`~astrogwb.sampling.amplitude.amplitude_statistics` and
:func:`~astrogwb.sampling.amplitude.best_fit_residual`. The numerical
marginalizer therefore consumes the *same* sufficient statistics as the
analytic one and shares its signature, so the two are drop-in interchangeable
via :class:`~astrogwb.sampling.protocol.LogEvidenceFn`. It also inherits the
high-SNR cancellation safety of ``R`` computed from residuals.

Folding the trapezoid weights into a precomputed log-measure turns the
integral over the grid into a single ``logsumexp``. With :math:`w_k` the
trapezoid weights at grid node :math:`\varphi_k`,

.. math::

    \ln Z = \ln \mathcal{N}_d - R
        + \operatorname{logsumexp}_k\!\left[\ell_k - \tfrac{1}{2}\bigl(\rho(f_k - \hat{A})\bigr)^2\right]
        - \ln \Pi,

.. math::

    \ell_k = \ln\pi(\varphi_k) + \ln w_k, \qquad
    \ln \Pi = \operatorname{logsumexp}_k\, \ell_k,

where :math:`f_k = f(\varphi_k)` and :math:`\ln \Pi` normalizes the prior mass
on the grid, so ``log_prior`` may be passed unnormalized. Squaring
:math:`\rho(f_k - \hat{A})` rather than forming :math:`\rho^2(f_k-\hat A)^2`
avoids overflowing :math:`\rho^2` at very high SNR.

"Exact up to quadrature error" only holds if the grid resolves the conditional
posterior, whose width in :math:`\varphi` is :math:`\sigma_A/|f'(\varphi)|`.
No quadrature rule rescues a Gaussian bump spanning three nodes, so grid
adequacy must be checked with :func:`quadrature_effective_nodes`, not assumed.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from astrogwb.importance.diagnostics import relative_ess
from astrogwb.sampling.protocol import AmplitudeScalingFn


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
    """Scalar ``logsumexp(log_measure)``; a coverage diagnostic, not used to
    normalize the evidence (that normalization already happens inside
    :func:`quadrature_log_evidence`)."""


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
        and :func:`quadrature_log_evidence` normalizes it away.
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


def _log_integrand(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    quadrature: AmplitudeQuadrature,
) -> jax.Array:
    """Shared ``(..., K)`` log-integrand behind all three public functions below.

    Routing :func:`quadrature_log_evidence`, :func:`draw_marginalized_parameter`,
    and :func:`quadrature_effective_nodes` through one implementation is what
    keeps them from drifting apart.
    """
    scaled_residual = jnp.expand_dims(template_optimal_snr, -1) * (
        quadrature.amplitude - jnp.expand_dims(amplitude_ml, -1)
    )
    return quadrature.log_measure - 0.5 * scaled_residual**2


def quadrature_log_evidence(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    quadrature: AmplitudeQuadrature,
    residual: jax.Array,
    log_norm: jax.Array,
) -> jax.Array:
    r"""Log marginal likelihood :math:`\ln\int p(\mathbf{d}\mid f(\varphi),\theta)\,\pi(\varphi)\,d\varphi`.

    ``LogEvidenceFn``-shaped: bind ``quadrature`` with :func:`functools.partial`
    to use as the ``log_evidence_fn`` argument of
    :func:`~astrogwb.sampling.models.amplitude_marginalized_model`.

    Parameters
    ----------
    amplitude_ml, template_optimal_snr:
        Sufficient statistics from
        :func:`~astrogwb.sampling.amplitude.amplitude_statistics`.
    quadrature:
        Precomputed grid from :func:`make_amplitude_quadrature`.
    residual:
        Best-fit residual :math:`R` from
        :func:`~astrogwb.sampling.amplitude.best_fit_residual`.
    log_norm:
        Gaussian normalization from
        :func:`~astrogwb.sampling.amplitude.gaussian_log_norm`.
    """
    log_integrand = _log_integrand(amplitude_ml, template_optimal_snr, quadrature)
    return (
        log_norm
        - residual
        + logsumexp(log_integrand, axis=-1)
        - quadrature.log_prior_mass
    )


def draw_marginalized_parameter(
    amplitude_ml: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    quadrature: AmplitudeQuadrature,
    rng_key: jax.Array,
) -> jax.Array:
    r"""Draw one value of :math:`\varphi` per posterior sample of :math:`\theta`.

    Inverse-transform sampling on the grid: the CDF is the cumulative sum of
    the same ``log_measure``-weighted nodes that :func:`quadrature_log_evidence`
    integrates, *not* a separately-computed cumulative trapezoid, so it
    terminates at exactly the :math:`Z` that was marginalized and the draws
    follow precisely that density rather than an :math:`O(\Delta^2)`-nearby
    one.

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
        Sufficient statistics from
        :func:`~astrogwb.sampling.amplitude.amplitude_statistics`.
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
    log_integrand = _log_integrand(amplitude_ml, template_optimal_snr, quadrature)
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
    :func:`quadrature_log_evidence` returns a finite, plausible-looking number;
    this should be comfortably above approximately 30.
    """
    log_integrand = _log_integrand(amplitude_ml, template_optimal_snr, quadrature)
    return relative_ess(log_integrand) * quadrature.grid.shape[0]
