r"""Physical shot noise as a rank-one term of the spectrum's covariance.

The observed spectrum is one catalog realization: a finite Poisson sum over
mergers, which scatters about the expected spectrum :math:`\mathbf{S}` on top
of the detector noise :math:`D = \mathrm{diag}(\sigma^2)`. Modelling that
scatter as one Gaussian mode :math:`\mathbf{u}` with unit variance gives the
covariance :math:`C = D + \mathbf{u}\mathbf{u}^T`, whose inverse and
determinant follow from the Sherman-Morrison identity. With the whitened
residual :math:`x = (\mathbf{d} - \mathbf{S})/\sigma` and direction
:math:`v = \mathbf{u}/\sigma`,

.. math::

    \chi^2 = x^T x - \frac{(x^T v)^2}{1 + v^T v}, \qquad
    \ln\det C = \sum_f \ln\sigma_f^2 + \ln(1 + v^T v).

Two directions are offered:

- *Amplitude* (:func:`amplitude_direction`): :math:`\mathbf{u} = s\,\mathbf{S}`,
  a multiplicative nuisance :math:`1 + \epsilon` with an informative prior
  :math:`\epsilon \sim \mathcal{N}(0, s^2)`, marginalized. Along the template
  this is a Gaussian in the amplitude estimator with variance
  :math:`\rho^{-2} + A^2 s^2`; the shape residual is unchanged.
- *Per frequency* (:func:`per_frequency_direction`):
  :math:`u_f = \sqrt{V_f}`, the right variance in every bin and still perfectly
  correlated across bins. It equals the amplitude direction wherever
  :math:`\sqrt{V_f}/S_f` is flat, and is a cross-check where a parameter tilts
  the template.

:math:`V_f` is the per-bin variance a
:class:`~astrogwb.inference.protocol.SpectralVarianceFn` predicts, for
example :func:`~astrogwb.gwb.importance.build_rescaled_shot_noise`.

Everything broadcasts over leading dimensions and contracts over the trailing
frequency axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = [
    "amplitude_direction",
    "amplitude_shot_noise_variance",
    "per_frequency_direction",
    "rank_one_gaussian_log_likelihood",
]


def _keep(frequency_mask: ArrayLike | None, like: jax.Array) -> jax.Array:
    """The bins counted, broadcast against ``like``."""
    if frequency_mask is None:
        return jnp.ones(jnp.shape(like), dtype=bool)
    return jnp.broadcast_to(jnp.asarray(frequency_mask), jnp.shape(like))


def amplitude_shot_noise_variance(
    spectrum: ArrayLike,
    variance: ArrayLike,
    scale: ArrayLike,
    frequency_mask: ArrayLike | None = None,
) -> jax.Array:
    r"""The relative amplitude variance :math:`s^2` of the shot noise.

    .. math::

        s = \frac{\sum_f w_f \sqrt{V_f} / S_f}{\sum_f w_f}, \qquad
        w_f = \frac{S_f^2}{\sigma_f^2},

    the SNR-weighted mean relative standard deviation over the counted bins
    with signal. It is exact when the shot noise is perfectly correlated
    across bins, and otherwise an upper bound on the variance of the
    amplitude estimator (Cauchy-Schwarz).

    Parameters
    ----------
    spectrum, variance:
        The expected spectrum :math:`S` and its per-bin shot-noise variance
        :math:`V`, shape ``(..., F)``.
    scale:
        Per-bin detector standard deviation :math:`\sigma`, ``(F,)``.
    frequency_mask:
        Boolean ``(F,)`` of the bins counted; ``None`` counts all.

    Returns
    -------
    jax.Array
        :math:`s^2`, shape ``(...)``.
    """
    spectrum = jnp.asarray(spectrum)
    variance = jnp.asarray(variance)
    usable = _keep(frequency_mask, spectrum) & (spectrum > 0.0)
    safe_spectrum = jnp.where(usable, spectrum, 1.0)
    weight = jnp.where(usable, (safe_spectrum / jnp.asarray(scale)) ** 2, 0.0)
    ratio = jnp.sqrt(jnp.where(usable, variance, 0.0)) / safe_spectrum
    scatter = jnp.sum(weight * ratio, axis=-1) / jnp.sum(weight, axis=-1)
    return scatter**2


def amplitude_direction(spectrum: ArrayLike, relative_variance: ArrayLike) -> jax.Array:
    r""":math:`\mathbf{u} = s\,\mathbf{S}`, the shot noise as an amplitude mode.

    ``relative_variance`` is :math:`s^2`, shape ``(...)``, from
    :func:`amplitude_shot_noise_variance` at the same point or fixed.
    """
    return jnp.sqrt(jnp.asarray(relative_variance))[..., None] * jnp.asarray(spectrum)


def per_frequency_direction(variance: ArrayLike) -> jax.Array:
    r""":math:`u_f = \sqrt{V_f}`, the shot noise with its per-bin variance."""
    return jnp.sqrt(jnp.asarray(variance))


def rank_one_gaussian_log_likelihood(
    observed: ArrayLike,
    prediction: ArrayLike,
    scale: ArrayLike,
    direction: ArrayLike,
    frequency_mask: ArrayLike | None = None,
) -> jax.Array:
    r"""Gaussian log-likelihood with covariance :math:`D + \mathbf{u}\mathbf{u}^T`.

    Evaluated with the Sherman-Morrison identity on whitened quantities, so it
    stays finite for per-bin scales many orders of magnitude from one. A zero
    ``direction`` gives the diagonal Gaussian exactly.

    Parameters
    ----------
    observed, prediction:
        Data and expected spectrum, ``(..., F)``.
    scale:
        Per-bin detector standard deviation, ``(F,)``.
    direction:
        The shot-noise mode :math:`\mathbf{u}`, ``(..., F)``.
    frequency_mask:
        Boolean ``(F,)`` of the bins counted; ``None`` counts all. Excluded
        bins contribute exactly zero, so the result equals evaluating on the
        arrays compressed to the selection, and a masked bin's ``scale`` may
        be infinite.

    Returns
    -------
    jax.Array
        Shape ``(...)``.
    """
    scale = jnp.asarray(scale)
    residual = jnp.asarray(observed) - jnp.asarray(prediction)
    direction = jnp.asarray(direction)
    residual, direction = jnp.broadcast_arrays(residual, direction)
    keep = _keep(frequency_mask, residual)
    safe_scale = jnp.where(keep, scale, 1.0)
    whitened = jnp.where(keep, residual / safe_scale, 0.0)
    mode = jnp.where(keep, direction / safe_scale, 0.0)
    mode_norm = jnp.sum(mode * mode, axis=-1)
    projection = jnp.sum(whitened * mode, axis=-1)
    chi_squared = jnp.sum(whitened * whitened, axis=-1) - projection**2 / (
        1.0 + mode_norm
    )
    log_scale = jnp.where(keep, jnp.log(safe_scale) + 0.5 * jnp.log(2.0 * jnp.pi), 0.0)
    return -0.5 * chi_squared - jnp.sum(log_scale, axis=-1) - 0.5 * jnp.log1p(mode_norm)
