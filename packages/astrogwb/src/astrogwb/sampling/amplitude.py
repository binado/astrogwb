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
than requiring the prior to be stated on :math:`A` itself. The log-integrand
on the grid is

.. math::

    \ell(\varphi) = \ln\pi(\varphi) - \tfrac{1}{2}\bigl(\rho\bigl(f(\varphi) - \hat{A}\bigr)\bigr)^2,

and the amplitude direction is integrated with the trapezoid rule after a
stable max-shift:

.. math::

    \ln Z = \ln p(d \mid \hat{A})
        + \ell_{\max}
        + \ln\!\int \exp\bigl(\ell(\varphi) - \ell_{\max}\bigr)\, d\varphi,

where :math:`\ln p(d \mid \hat{A}) = \ln\mathcal{N}_d - R` is the Gaussian
log-likelihood at the MLE amplitude. The caller must supply a prior density
that is already normalized on the grid
(:math:`\int \pi(\varphi)\, d\varphi \approx 1` under the same trapezoid
rule); this module does not renormalize. Squaring
:math:`\rho(f(\varphi) - \hat{A})` rather than forming
:math:`\rho^2(f-\hat A)^2` avoids overflowing :math:`\rho^2` at very high
SNR.

"Exact up to quadrature error" only holds if the grid resolves the conditional
posterior, whose width in :math:`\varphi` is :math:`\sigma_A/|f'(\varphi)|`.
No quadrature rule rescues a Gaussian bump spanning three nodes, so grid
adequacy must be checked with :func:`quadrature_effective_nodes`, not assumed.

All functions broadcast over leading batch dimensions and contract over the
trailing frequency axis, so post-processing can feed them ``(chain, draw)``
shaped arrays directly.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from astrogwb.importance.diagnostics import relative_ess


class AmplitudeScalingFn(Protocol):
    """Map the marginalized parameter to the multiplicative amplitude :math:`A = f(\\varphi)`."""

    def __call__(self, marginalized_parameter: jax.Array) -> jax.Array: ...


class AmplitudeQuadrature(NamedTuple):
    """Precomputed grid, scaling, and prior for numerical marginalization.

    Built once by :func:`make_amplitude_quadrature` and then reused every MCMC
    step; a plain ``NamedTuple`` keeps it a JAX pytree without needing to be
    stored as model state, since the callable that built it is only consumed
    at construction time.
    """

    grid: jax.Array
    """``(K,)`` values of the marginalized parameter :math:`\\varphi`."""

    amplitude: jax.Array
    """``(K,)`` :math:`f(\\varphi_k)`, the multiplicative factor at each node."""

    log_prior: jax.Array
    """``(K,)`` :math:`\\ln\\pi(\\varphi_k)`, the caller's prior density on the grid."""


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
        Must already be normalized on the grid
        (:math:`\\int \\pi(\\varphi)\\, d\\varphi \\approx 1` under the
        trapezoid rule); typical source is a NumPyro
        ``Distribution.log_prob`` evaluated on a grid that covers the prior
        support. This factory does not renormalize.
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
    if not bool(jnp.all(jnp.diff(grid) > 0)):
        raise ValueError("grid must be strictly increasing")

    return AmplitudeQuadrature(
        grid=grid,
        amplitude=scaling(grid),
        log_prior=log_prior,
    )


def amplitude_log_integrand(
    amplitude_mle: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    quadrature: AmplitudeQuadrature,
) -> jax.Array:
    r"""Shared ``(..., K)`` log-integrand behind the model factor and the two
    functions below.

    Returns :math:`\ln\pi(\varphi_k) - \tfrac12[\rho(f_k - \hat A)]^2` -- the
    continuous density on the grid, before trapezoid integration.

    Routing :func:`~astrogwb.sampling.models.amplitude_marginalized_model`,
    :func:`draw_marginalized_parameter`, and :func:`quadrature_effective_nodes`
    through one implementation is what keeps them from drifting apart -- in
    particular, it guarantees the inverse-transform sampler in
    :func:`draw_marginalized_parameter` draws from exactly the density the
    model factor integrated.
    """
    scaled_residual = jnp.expand_dims(template_optimal_snr, -1) * (
        quadrature.amplitude - jnp.expand_dims(amplitude_mle, -1)
    )
    return quadrature.log_prior - 0.5 * scaled_residual**2


def log_trapezoid(log_y: jax.Array, x: jax.Array) -> jax.Array:
    r"""Stable :math:`\ln\int \exp(\log y)\, dx` via a shifted trapezoid rule.

    Broadcasts over leading dimensions of ``log_y`` and integrates along the
    trailing axis against the 1D abscissa ``x``.
    """
    log_y_max = jnp.max(log_y, axis=-1, keepdims=True)
    integral = jnp.trapezoid(jnp.exp(log_y - log_y_max), x, axis=-1)
    return jnp.squeeze(log_y_max, axis=-1) + jnp.log(integral)


def _cumulative_trapezoid(y: jax.Array, x: jax.Array) -> jax.Array:
    """Cumulative trapezoid integral of ``y`` vs ``x``, starting at 0."""
    dx = jnp.diff(x)
    segments = 0.5 * (y[..., :-1] + y[..., 1:]) * dx
    zeros = jnp.zeros(y.shape[:-1] + (1,), dtype=y.dtype)
    return jnp.concatenate([zeros, jnp.cumsum(segments, axis=-1)], axis=-1)


def draw_marginalized_parameter(
    amplitude_mle: jax.Array,
    template_optimal_snr: jax.Array,
    *,
    quadrature: AmplitudeQuadrature,
    rng_key: jax.Array,
) -> jax.Array:
    r"""Draw one value of :math:`\varphi` per posterior sample of :math:`\theta`.

    Inverse-transform sampling on the grid: the CDF is the cumulative trapezoid
    of the same integrand that :func:`log_trapezoid` integrates in the model
    factor, so the draws follow precisely the density that was marginalized.

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
    amplitude_mle, template_optimal_snr:
        The amplitude sufficient statistics, as computed by
        :func:`~astrogwb.sampling.models.amplitude_marginalized_model`.
    quadrature:
        Precomputed grid from :func:`make_amplitude_quadrature`.
    rng_key:
        PRNG key; one uniform draw is consumed per leading-dimension element.

    Returns
    -------
    jax.Array
        :math:`\varphi` draws, same leading shape as ``amplitude_mle``, clipped
        to ``[grid[0], grid[-1]]``.
    """
    log_integrand = amplitude_log_integrand(
        amplitude_mle, template_optimal_snr, quadrature=quadrature
    )
    shifted = jnp.exp(log_integrand - jnp.max(log_integrand, axis=-1, keepdims=True))
    cdf = _cumulative_trapezoid(shifted, quadrature.grid)
    cdf = cdf / cdf[..., -1:]

    grid = quadrature.grid
    num_nodes = grid.shape[0]
    u = jax.random.uniform(rng_key, shape=amplitude_mle.shape)
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
    amplitude_mle: jax.Array,
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
        amplitude_mle, template_optimal_snr, quadrature=quadrature
    )
    return relative_ess(log_integrand) * quadrature.grid.shape[0]
