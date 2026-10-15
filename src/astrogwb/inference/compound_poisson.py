r"""The exact density of a compound-Poisson amplitude in Gaussian noise.

A catalog realization projected on a template is a compound-Poisson sum: the
point set :math:`\{(a_i, \beta_i)\}` holds expected counts :math:`a_i` of
sources each contributing :math:`\beta_i`, so the amplitude estimator is

.. math::

    \hat B = \sum_i N_i \beta_i + n, \qquad
    N_i \sim \mathrm{Poisson}(a_i), \qquad n \sim \mathcal{N}(0, \rho^{-2}),

with cumulant generating function, for real or complex :math:`t`,

.. math::

    K(t) = \sum_i a_i \left(e^{t \beta_i} - 1\right) + \frac{t^2}{2 \rho^2}.

The density is the Bromwich integral
:math:`p(b) = \frac{1}{2\pi} \int e^{K(t + iy) - (t + iy) b}\, dy` along any
vertical line. On the line through the saddle :math:`K'(t) = b` the integrand
is a real Gaussian in :math:`y` times a smooth factor, so

- :func:`hermite_log_density` integrates it with Gauss-Hermite nodes, exact as
  the node count grows;
- :func:`saddlepoint_log_density` is its one-node limit, renormalized;
- :func:`gaussian_log_density` keeps the first two cumulants only, the
  Gaussian approximation the rank-one likelihoods of
  :mod:`astrogwb.inference.shot_noise` make.

Every :math:`\beta_i` must be bounded for :math:`K` to exist on the real line;
a redshift window bounded away from zero bounds them. Sums run over every
axis of the point set, so any shape broadcasts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = [
    "MAX_NEWTON_STEPS",
    "SADDLE_TOLERANCE",
    "AmplitudeDensityMethod",
    "PointSet",
    "amplitude_log_density",
    "cumulants",
    "gaussian_log_density",
    "hermite_log_density",
    "inclination_quadrature",
    "saddle",
    "saddlepoint_log_density",
]

#: How :func:`amplitude_log_density` evaluates :math:`p(\hat B)`.
type AmplitudeDensityMethod = Literal["saddlepoint", "hermite", "gaussian"]

#: Newton on the saddle stops once a step is below this many standard
#: deviations of the tilted density.
SADDLE_TOLERANCE = 1e-10

#: The most Newton steps :func:`saddle` takes by default.
MAX_NEWTON_STEPS = 64


class PointSet(NamedTuple):
    r"""A compound-Poisson amplitude in Gaussian noise.

    Attributes
    ----------
    weights:
        Expected counts :math:`a_i \geq 0`, any shape.
    amplitudes:
        Per-source contributions :math:`\beta_i`, the shape of ``weights``.
    noise_precision:
        :math:`\rho^2`, the inverse variance of the Gaussian noise, ``()``.
    """

    weights: jax.Array
    amplitudes: jax.Array
    noise_precision: jax.Array


def inclination_quadrature(num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Gauss-Legendre nodes for the inclination average of the power.

    A source's power scales with
    :math:`\tilde F(c) = (1 + 6 c^2 + c^4) / 3.2` in :math:`c = \cos\iota`,
    uniform on :math:`[0, 1]`, normalized to unit mean. ``num_nodes`` nodes
    integrate polynomials of degree :math:`2 n - 1` exactly, so from three on
    the mean and from five on the second moment
    (:data:`~astrogwb.gwb.importance.INCLINATION_SECOND_MOMENT`) are exact.

    Returns
    -------
    tuple of numpy.ndarray
        :math:`\tilde F(c_l)` and the weights :math:`u_l` (summing to one),
        each ``(num_nodes,)``. Host constants, for a static ``num_nodes``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(num_nodes)
    cosine = 0.5 * (nodes + 1.0)
    return (1.0 + 6.0 * cosine**2 + cosine**4) / 3.2, 0.5 * weights


def cumulants(t: ArrayLike, points: PointSet) -> tuple[jax.Array, ...]:
    r""":math:`K(t)`, :math:`K'(t)` and :math:`K''(t)` at real ``t``, each ``()``.

    The derivatives are explicit sums, not automatic derivatives of :math:`K`,
    so differentiating a function of them is first-order AD.
    """
    t = jnp.asarray(t)
    weights, amplitudes, precision = points
    # One transcendental per point: the tilt is expm1 plus one.
    excess = weights * jnp.expm1(t * amplitudes)
    tilted = weights + excess
    value = jnp.sum(excess) + 0.5 * t**2 / precision
    first = jnp.sum(tilted * amplitudes) + t / precision
    second = jnp.sum(tilted * amplitudes**2) + 1.0 / precision
    return value, first, second


def _origin_moments(points: PointSet) -> tuple[jax.Array, jax.Array]:
    """:math:`K'(0)` and :math:`K''(0)`, the mean and variance."""
    weights, amplitudes, precision = points
    return (
        jnp.sum(weights * amplitudes),
        jnp.sum(weights * amplitudes**2) + 1.0 / precision,
    )


def saddle(
    b: ArrayLike, points: PointSet, max_newton_steps: int = MAX_NEWTON_STEPS
) -> jax.Array:
    r"""The saddle :math:`t` with :math:`K'(t) = b`, ``()``.

    Newton's method from the Gaussian saddle :math:`t_0 = (b - K'(0)) / K''(0)`.
    For nonnegative :math:`\beta` the derivative :math:`K'` is increasing and
    convex, so :math:`K'(t_0) \geq b` and the iterates decrease monotonically
    onto the root: in two or three steps near the Gaussian limit, in more far
    in the right tail, where each step moves by about :math:`1/\max\beta`. It
    stops once a step is below :data:`SADDLE_TOLERANCE` standard deviations of
    the tilted density, or after ``max_newton_steps`` (static). The result is a
    :func:`jax.lax.custom_root`: derivatives of every order follow from the
    implicit function theorem, not from the iterations.
    """
    b = jnp.asarray(b)
    mean, variance = _origin_moments(points)

    def residual(t: jax.Array) -> jax.Array:
        return cumulants(t, points)[1] - b

    def newton(_: object, initial: jax.Array) -> jax.Array:
        def step(state: tuple[jax.Array, jax.Array, jax.Array]) -> tuple:
            t, _, count = state
            _, first, second = cumulants(t, points)
            return t - (first - b) / second, (first - b) / jnp.sqrt(second), count + 1

        def unconverged(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
            _, standardized_step, count = state
            return (jnp.abs(standardized_step) > SADDLE_TOLERANCE) & (
                count < max_newton_steps
            )

        start = (initial, jnp.asarray(jnp.inf, initial.dtype), jnp.asarray(0))
        return jax.lax.while_loop(unconverged, step, start)[0]

    def tangent_solve(
        linear: Callable[[jax.Array], jax.Array], rhs: jax.Array
    ) -> jax.Array:
        return rhs / linear(jnp.ones_like(rhs))

    return jax.lax.custom_root(residual, (b - mean) / variance, newton, tangent_solve)


def _over_nodes(fn: Callable[[jax.Array], jax.Array], nodes: np.ndarray) -> jax.Array:
    """``fn`` at each quadrature node, one node at a time, ``(n,)``.

    Each evaluation is a pass over the whole point set; mapping sequentially
    and rematerializing in the backward pass keeps one pass's intermediates
    alive instead of every node's.
    """
    return jax.lax.map(jax.checkpoint(fn), jnp.asarray(nodes))


def gaussian_log_density(b: ArrayLike, points: PointSet) -> jax.Array:
    r"""The Gaussian approximation :math:`\mathcal{N}(b; K'(0), K''(0))`, ``()``."""
    mean, variance = _origin_moments(points)
    return -0.5 * (jnp.asarray(b) - mean) ** 2 / variance - 0.5 * jnp.log(
        2.0 * jnp.pi * variance
    )


def saddlepoint_log_density(
    b: ArrayLike,
    points: PointSet,
    num_normalization_nodes: int = 4,
) -> jax.Array:
    r"""The normalized saddlepoint approximation of :math:`\log p(b)`, ``()``.

    .. math::

        \log p(b) = K(t) - t b - \tfrac12 \log\left(2\pi K''(t)\right) - \log Z,
        \qquad K'(t) = b.

    :math:`Z` integrates the unnormalized density over :math:`b = K'(t)`,

    .. math::

        Z = \int e^{K(t) - t K'(t)} \sqrt{\frac{K''(t)}{2\pi}}\, dt,

    with ``num_normalization_nodes`` Gauss-Hermite nodes at
    :math:`t_k = \sqrt{2 / K''(0)}\, u_k`, so it needs no further saddles and
    does not depend on ``b``.
    """
    b = jnp.asarray(b)
    t = saddle(b, points)
    value, _, second = cumulants(t, points)
    unnormalized = value - t * b - 0.5 * jnp.log(2.0 * jnp.pi * second)

    nodes, node_weights = np.polynomial.hermite.hermgauss(num_normalization_nodes)
    width = jnp.sqrt(2.0 / _origin_moments(points)[1])

    def integrand(u: jax.Array) -> jax.Array:
        tk = width * u
        k0, k1, k2 = cumulants(tk, points)
        return k0 - tk * k1 + u**2 + 0.5 * jnp.log(k2 / (2.0 * jnp.pi))

    log_normalization = jax.scipy.special.logsumexp(
        _over_nodes(integrand, nodes), b=jnp.asarray(node_weights)
    ) + jnp.log(width)
    return unnormalized - log_normalization


def hermite_log_density(
    b: ArrayLike,
    points: PointSet,
    num_nodes: int = 8,
) -> jax.Array:
    r"""Gauss-Hermite quadrature of the Bromwich integral through the saddle, ``()``.

    With :math:`z = t + iy` and :math:`K'(t) = b`,

    .. math::

        p(b) = \frac{e^{K(t) - t b}}{2\pi} \sqrt{\frac{2}{K''(t)}}
               \sum_k \omega_k\, \mathrm{Re}\, h(y_k), \qquad
        y_k = \sqrt{\frac{2}{K''(t)}}\, u_k,

    where :math:`h(y) = e^{K(z) - z b - K(t) + t b + K''(t) y^2 / 2}` is one
    plus terms of order :math:`(y\beta)^3`. Its modulus and phase are summed
    in forms with no cancellation between terms of size :math:`\sum a`:

    .. math::

        \log|h| = \sum a e^{t\beta}
                  \left(\tfrac12 (y\beta)^2 - 2 \sin^2\tfrac{y\beta}{2}\right),
        \qquad
        \arg h = y \left(K'(t) - b\right) + \sum a e^{t\beta} (\sin y\beta - y\beta).

    A point set far from Gaussian (few sources, little noise) may need many
    nodes, and too few can make the sum negative and the result ``nan``.
    """
    b = jnp.asarray(b)
    weights, amplitudes, _ = points
    t = saddle(b, points)
    value, first, second = cumulants(t, points)
    tilted = weights * jnp.exp(t * amplitudes)
    width = jnp.sqrt(2.0 / second)

    def real_part(u: jax.Array) -> jax.Array:
        phase_arg = width * u * amplitudes
        log_modulus = jnp.sum(
            tilted * (0.5 * phase_arg**2 - 2.0 * jnp.sin(0.5 * phase_arg) ** 2)
        )
        phase = width * u * (first - b) + jnp.sum(
            tilted * (jnp.sin(phase_arg) - phase_arg)
        )
        return jnp.exp(log_modulus) * jnp.cos(phase)

    nodes, node_weights = np.polynomial.hermite.hermgauss(num_nodes)
    total = jnp.asarray(node_weights) @ _over_nodes(real_part, nodes)
    return value - t * b + jnp.log(total * width / (2.0 * jnp.pi))


def amplitude_log_density(
    b: ArrayLike,
    points: PointSet,
    method: AmplitudeDensityMethod = "saddlepoint",
    *,
    num_hermite_nodes: int = 8,
    num_normalization_nodes: int = 4,
) -> jax.Array:
    """:math:`\\log p(b)` of ``points`` by ``method`` (static), ``()``.

    ``num_hermite_nodes`` is the contour quadrature of ``"hermite"``,
    ``num_normalization_nodes`` the normalization of ``"saddlepoint"``.
    """
    if method == "saddlepoint":
        return saddlepoint_log_density(b, points, num_normalization_nodes)
    if method == "hermite":
        return hermite_log_density(b, points, num_hermite_nodes)
    return gaussian_log_density(b, points)
